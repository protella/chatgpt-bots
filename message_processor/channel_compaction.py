"""The compaction builder: a channel's history → one published snapshot (plan §1e–§1n).

Everything here is derived. The crawl walks Slack, keeps an event SKELETON of identities, ranks,
byte counts and fingerprints, and throws every fetched message away at the end of the chunk that
used it. No transcript is persisted, by schema shape rather than by care.

The three public entry points the coordinator drives are `run_crawl_slice`, `run_incremental` and
`publish_stale_retained`; everything above them is a pure function so the arithmetic that decides
a boundary, a chunk digest or a byte cap can be tested without a database or a network.

OFFLINE-MUTATION RESIDUAL (§1c). Sources are fingerprinted during a crawl (`projection_sha256`),
so an edit that happened BEFORE or DURING a crawl is caught there and takes the discard path. What
remains uncovered is specifically **a mutation occurring AFTER PUBLICATION while the process is
offline**: Slack delivered the edit while we were down, so no observation row was written for it,
and nothing later re-reads the span to notice.

That residual has ONE SIBLING, and it is documented at its own site rather than repeated here:
`feed_own_mutation` (slack_client/event_handlers/activity_index.py) carries no admission ticket,
so a failed owned-operation observation is never retried. It collapses INTO the residual above —
the delivery feed covers the same edit whenever the socket is up and repairs its own write
failures through its ticket, so losing the invalidation outright requires the delivery never to
have arrived, which means the process was effectively offline for that mutation.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set,
                    Tuple)

from config import config
from logger import setup_logger
from slack_client.history_fetch import HistoryFetchError, PageResult, fetch_page
from slack_client.normalizer import (
    NormalizedMessage,
    ORIGIN_HISTORY,
    ORIGIN_REPLIES,
    TimestampError,
    in_window,
    normalize_slack_message,
    parse_ts,
    render_mentions,
    sanitize_name,
)
from token_counter import ITEM_STRUCTURAL_OVERHEAD, admission_charge

logger = setup_logger(name="slack_bot.ChannelCompaction")

# `prompt_version` covers BOTH prompt constants AND the B1 projection grammar. Changing either
# changes this string, which discards live crawl checkpoints (§1n step 2).
PROMPT_VERSION = "v1"

COMPACTION_MAP_PROMPT_V1 = (
    "Condense this slice of a Slack channel's history into factual notes. Preserve: who asked "
    "what and whether it was answered, decisions, open questions and who owns them, artifact "
    "references by their [artifact:...] tags, thread structure. Never invent, never editorialize, "
    "never address anyone. Output plain prose notes."
)

COMPACTION_REDUCE_PROMPT_V1 = (
    "Merge these notes into ONE condensed account of the channel's earlier history, "
    "\u2264{budget} tokens. Same preservation rules. Plain prose, past tense, no headers, no "
    "instructions, no address to the reader."
)

# A HASH chunk is exactly this many sealed skeleton events; the last chunk holds the remainder.
# Fixed so `source_hash` is reproducible. MAP chunks are a different thing entirely (§1g).
HASH_CHUNK_EVENTS = 500
SOURCE_HASH_DOMAIN = b"source-hash-v2"

PRIOR_SUMMARY_HEADER = "PRIOR SUMMARY (through {boundary_ts}):"
RENDER_LINE_PREFIX = "    [render:{namespace}:{row_id}] "

ROOT_SENTINEL = "0"
# Closed integer rank. Ranks are compared, never kind strings (§1n).
KIND_RANK_MESSAGE = 0
KIND_RANK_REPLY = 1
KIND_RANK_TOMBSTONE = 2

SOURCE_RANK_HISTORY = 1
SOURCE_RANK_REPLIES = 2

CRAWL_MODE_RAW = "raw"
CRAWL_MODE_INCREMENTAL = "incremental"

PHASE_INVENTORY = 1
PHASE_CHUNKS = 2

# §1e receipt proof states, frozen at the source pin. `in_flight` is valid evidence for the
# BOUNDARY CAP and is never proof of summarizable inclusion.
PROOF_FINALIZED = "finalized"
PROOF_CHROME = "chrome_excluded"
PROOF_GRANDFATHERED = "pre_epoch_grandfathered"
PROOF_IN_FLIGHT = "in_flight"
PROOF_UNPROVEN = "unproven"
SUMMARIZABLE_PROOFS = frozenset({PROOF_FINALIZED, PROOF_GRANDFATHERED})

ANCHOR_AVAILABLE = "available"
ANCHOR_UNAVAILABLE = "unavailable"
ANCHOR_REFUSED = "refused"
ANCHOR_UNSAFE = "unsafe"
RECEIPT_PROOF_NOT_SELF = "not_self"

BUILD_OK = "ok"
BUILD_FAILED = "failed"
BUILD_DISCARDED = "discarded"
BUILD_COPIED = "copied"

FIT_UNDER_TARGET = "under_target"
FIT_UNDER_TRIGGER = "under_trigger"
FIT_NONE = "no_fit"

# The step-2 reset list (§1n). A difference in ANY of these discards progress; none of them
# increments `consecutive_discards`, and none of them clears an active backoff.
RESET_FIELDS: Tuple[str, ...] = (
    "serializer_version", "serializer_config_hash", "prompt_version", "sizing_profile",
    "profile_version", "actor_snapshot_hash", "crawl_mode", "input_floor_ts", "source_floor_ts",
)

STALL_DISCARD_THRESHOLD = 3
STALL_BACKOFF_SECONDS = 3600.0


class CompactionError(Exception):
    """A compaction attempt cannot proceed. Never raised at a caller that has a turn to serve."""


class SourceMutated(CompactionError):
    """Phase II found a mutation: an unmatched event, an unmatched skeleton row, or a
    fingerprint mismatch. Takes the §1n step-3 discard path."""

    def __init__(self, message: str, *, subject_ts: Optional[str] = None):
        super().__init__(message)
        self.subject_ts = subject_ts


# ---------------------------------------------------------------- A2 render seam
#
# Appendix A bytes are produced in exactly ONE place — `channel_stream`. These are thin
# delegations, deliberately with NO local fallback: a stand-in that rendered "close enough" bytes
# would drift from the serializer silently, and the drift would only surface as a summary that
# does not fit the room it was sized against.

def _stream_helper(name: str) -> Callable[..., Any]:
    from message_processor import channel_stream

    helper = getattr(channel_stream, name, None)
    if not callable(helper):
        raise CompactionError(f"channel_stream.{name} is missing; serializer v2 is incomplete")
    return helper


def _escape_payload(text: str) -> str:
    """A7 over a whole payload. Idempotent, so escaping before validation and letting the
    renderer escape again changes nothing."""
    return _stream_helper("escape_payload")(text)


def render_anchor_block(anchors: Sequence[Mapping[str, Any]], *, omitted: int = 0) -> str:
    return _stream_helper("render_anchor_block")(anchors, omitted=omitted)


def render_summary_block(*, boundary_ts: str, payload: str, anchor_block: str,
                         stale: bool = False) -> str:
    return _stream_helper("render_summary_block")(
        boundary_ts=boundary_ts, payload=payload, anchor_block=anchor_block, stale=stale)


def stale_marked_payload(payload_bytes: bytes, *, boundary_ts: str) -> bytes:
    """§1f copy semantics: EXACTLY ONE stale marker, ever."""
    return _stream_helper("stale_marked_payload")(payload_bytes, boundary_ts=boundary_ts)


def _root_snippet_bytes(root: Optional[NormalizedMessage]) -> int:
    """The EXACT rendered suffix bytes of the v1 reply header's root snippet.

    Sanitization, the word/char caps, multi-byte characters, the empty-text case and the
    tombstone's fixed marker are all inside `_root_snippet`; raw text length is wrong in every
    one of them.
    """
    if root is None:
        return 0
    return len(_stream_helper("_root_snippet")(root).encode("utf-8"))


# ---------------------------------------------------------------- Appendix B1 projection

# Everything below \x20 plus DEL. \n and \t are already literalized by the time this runs.
_PROJECTION_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_PROJECTION_SENDER = {"human": "human", "other_bot": "bot", "self": "self"}


def projection_escape(text: Any) -> str:
    """Projection escaping (B1) — deliberately NOT the A7 rule.

    A newline becomes the LITERAL two characters `\\n` and a tab `\\t`; every other control
    character is stripped. That is what makes "one physical line per event" hold; A7 preserves
    real newlines and would break the grammar here.
    """
    value = str(text if text is not None else "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\n", "\\n").replace("\t", "\\t")
    return _PROJECTION_CONTROL_RE.sub("", value)


def kind_rank(message: NormalizedMessage) -> int:
    if message.is_tombstone:
        return KIND_RANK_TOMBSTONE
    return KIND_RANK_REPLY if message.is_reply else KIND_RANK_MESSAGE


def root_key(message: NormalizedMessage) -> str:
    """The ordering key's root field. Top-level is the sentinel `"0"`, never None: comparing
    None with a string is not safely orderable."""
    return str(message.thread_root_ts) if message.is_reply else ROOT_SENTINEL


def order_key(message: NormalizedMessage) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """The composite total order `(ts, root_ts, kind_rank)` after dedup."""
    root = root_key(message)
    return (parse_ts(message.ts), (0, 0) if root == ROOT_SENTINEL else parse_ts(root),
            kind_rank(message))


def identity_key(message: NormalizedMessage) -> Tuple[str, int]:
    """Event identity — `(ts, kind_rank)`, deliberately WITHOUT root_ts so a broadcast's two
    copies are one row."""
    return (str(message.ts), kind_rank(message))


def project_event(message: NormalizedMessage, *,
                  actor_names: Optional[Mapping[str, str]] = None,
                  artifact_renders: Sequence[Mapping[str, Any]] = ()) -> str:
    """One event's Appendix B1 projection: its physical line, plus one bundled physical line per
    captured artifact/marker render immediately after it.

    The bundled lines are not independent events — they contribute to this event's
    `projected_byte_len` and `projection_sha256` and get no skeleton row of their own.
    """
    marker = ""
    if message.is_reply:
        marker = f" \u21b3{message.thread_root_ts}"
    if message.is_tombstone:
        marker += " \u2298"
    text = render_mentions(message.text, dict(actor_names or {}))
    line = (f"{message.ts} {message.sender_id or 'unknown'} "
            f"[{_PROJECTION_SENDER.get(message.sender_type, 'human')}]{marker} "
            f":: {projection_escape(text)}")
    for ref in message.files:
        line += f" [file: {sanitize_name(ref.name) or 'file'} ({ref.mimetype})]"
    for entry in artifact_renders:
        line += (f" [artifact:{entry.get('artifact_namespace')}:{entry.get('row_id')}"
                 f":{str(entry.get('content_hash') or '')[:8]}]")
    lines = [line]
    for entry in artifact_renders:
        prefix = RENDER_LINE_PREFIX.format(namespace=entry.get("artifact_namespace"),
                                           row_id=entry.get("row_id"))
        lines.append(prefix + projection_escape(entry.get("render")))
    return "\n".join(lines)


def prior_summary_chunk(payload_bytes: bytes, *, old_boundary_ts: str) -> str:
    """Chunk index 0 of an incremental projection: the parent's PERSISTED payload bytes,
    verbatim, under the literal B1 header."""
    body = payload_bytes.decode("utf-8", errors="replace")
    return PRIOR_SUMMARY_HEADER.format(boundary_ts=old_boundary_ts) + "\n" + body


def event_fingerprint(projection: str) -> str:
    """`projection_sha256` — what catches a SAME-LENGTH edit that changes neither the identity
    nor either byte count."""
    return hashlib.sha256(projection.encode("utf-8")).hexdigest()


def chunk_bytes(lines: Sequence[str]) -> bytes:
    """A chunk's exact projection bytes: LF joins, and a TERMINAL newline on every chunk
    including the last. Without both pinned, two conforming implementations produce different
    digests from identical events."""
    return ("\n".join(lines) + "\n").encode("utf-8")


def chunk_digest(lines: Sequence[str]) -> bytes:
    """SHA-256 over the chunk's exact projection bytes, as 32 RAW bytes."""
    return hashlib.sha256(chunk_bytes(lines)).digest()


def finish_source_hash(digests: Sequence[Any]) -> str:
    """The versioned tree hash. Resumable from the persisted digests alone — a crawl that
    stopped after chunk 7 finishes the hash without re-reading chunks 0–6."""
    running = hashlib.sha256()
    running.update(SOURCE_HASH_DOMAIN)
    for digest in digests:
        raw = bytes.fromhex(digest) if isinstance(digest, str) else bytes(digest)
        if len(raw) != 32:
            raise CompactionError(f"chunk digest is {len(raw)} bytes, not 32")
        running.update(raw)
    return running.hexdigest()


def hash_chunks(projections: Sequence[str], *,
                size: int = HASH_CHUNK_EVENTS) -> List[List[str]]:
    """Split projected events into hash chunks of EXACTLY `size`; the final chunk holds the
    remainder. Adjacent chunks neither overlap nor gap."""
    return [list(projections[i:i + size]) for i in range(0, len(projections), size)]


# ---------------------------------------------------------------- sizing identity

def sizing_profile(*, model: str, window: int, trigger_tokens: int,
                   target_tokens: int) -> str:
    """`"<model>:<window>:<trigger_tokens>:<target_tokens>"`, integers so the string is stable
    across float formatting. Trigger and target are IN the key: a publication proved to fit under
    a looser target says nothing about the stricter one that raised the obligation."""
    return f"{model}:{int(window)}:{int(trigger_tokens)}:{int(target_tokens)}"


def profile_version(*, headroom_source: str, headroom_tokens: int) -> str:
    return f"{headroom_source}:{int(headroom_tokens)}"


def resolve_sizing(*, model: str, window: int,
                   trigger_ratio: Optional[float] = None,
                   target_ratio: Optional[float] = None) -> Dict[str, Any]:
    trigger_ratio = (config.compaction_trigger_ratio if trigger_ratio is None
                     else trigger_ratio)
    target_ratio = (config.compaction_target_ratio if target_ratio is None else target_ratio)
    trigger_tokens = int(window * trigger_ratio)
    target_tokens = int(window * target_ratio)
    return {
        "model": model, "window": int(window),
        "trigger_tokens": trigger_tokens, "target_tokens": target_tokens,
        "sizing_profile": sizing_profile(model=model, window=window,
                                         trigger_tokens=trigger_tokens,
                                         target_tokens=target_tokens),
    }


def utility_map_bound(*, model: Optional[str] = None,
                      output_reserve: Optional[int] = None) -> int:
    """The admission bound one MAP or REDUCE call's input must stay under."""
    name = model or config.utility_model
    window = config.get_model_token_limit(name)
    reserve = int(config.summary_byte_cap if output_reserve is None else output_reserve)
    return max(1, int(window) - reserve)


# ---------------------------------------------------------------- map / reduce chunking

def subchunk(projections: Sequence[str], *, bound: int,
             base_charge: int = 0) -> List[List[str]]:
    """Deterministic subchunking of one HASH chunk into MAP chunks (§1g).

    Walk the projected events in order and close the current map chunk immediately BEFORE the
    first event that would carry its admission charge past the bound. A function of the
    projection bytes and the sizing profile alone, so it is reproducible.
    """
    chunks: List[List[str]] = []
    current: List[str] = []
    charge = base_charge
    for entry in projections:
        cost = admission_charge(entry) + 1  # the joining / terminal newline
        if current and charge + cost > bound:
            chunks.append(current)
            current = []
            charge = base_charge
        current.append(entry)
        charge += cost
    if current:
        chunks.append(current)
    return chunks


def map_chunks(hash_chunk_index: int, projections: Sequence[str], *, bound: int,
               base_charge: int = 0) -> List[Tuple[Tuple[int, int], List[str]]]:
    """Map chunks keyed `(hash_chunk_idx, sub_idx)` — NEVER index-parallel with the digests."""
    return [((hash_chunk_index, sub_idx), part)
            for sub_idx, part in enumerate(subchunk(projections, bound=bound,
                                                    base_charge=base_charge))]


def summary_key(raw: Any) -> Tuple[int, int]:
    """`"<chunk_idx>:<sub_idx>"` → the ordering pair. Persisted keys are strings because the
    checkpoint column is JSON."""
    if isinstance(raw, (tuple, list)):
        return (int(raw[0]), int(raw[1]))
    chunk, _, sub = str(raw).partition(":")
    return (int(chunk), int(sub or 0))


def ordered_summaries(chunk_summaries: Mapping[Any, str]) -> List[str]:
    """Map summaries in `(hash_chunk_idx, sub_idx)` key order — the reduce stage's input."""
    return [text for _key, text in
            sorted(chunk_summaries.items(), key=lambda item: summary_key(item[0]))]


def reduce_groups(summaries: Sequence[str], *, bound: int,
                  base_charge: int = 0) -> List[List[str]]:
    """One level of the HIERARCHICAL REDUCE: walk in key order, closing a group immediately
    before the first summary that would carry it past the bound."""
    return subchunk(summaries, bound=bound, base_charge=base_charge)


# ---------------------------------------------------------------- the summary byte cap

def summary_room_tokens(*, target_tokens: int, headroom_tokens: int, shell_bytes: int,
                        anchor_bytes: int, horizon_bytes: int, item_count: int,
                        retained_charge: int,
                        item_overhead: int = ITEM_STRUCTURAL_OVERHEAD) -> int:
    """The room a summary has, IN THE ADMITTED CURRENCY.

    Production admission charges ONE TOKEN PER UTF-8 BYTE, so one byte of summary consumes one
    token of this room. A bytes-per-token multiplier is not conservative here, it is wrong by
    that multiplier.
    """
    spent = (admission_charge_bytes(shell_bytes) + admission_charge_bytes(anchor_bytes)
             + admission_charge_bytes(horizon_bytes)
             + max(0, int(item_count)) * int(item_overhead)
             + max(0, int(retained_charge)) + max(0, int(headroom_tokens)))
    return max(0, int(target_tokens) - spent)


def admission_charge_bytes(byte_count: int) -> int:
    """One token per UTF-8 byte — the identity the admission currency makes true."""
    return max(0, int(byte_count))


def summary_shell_bytes(*, boundary_ts: str, stale: bool = False) -> int:
    """The A2 shell's own bytes, MEASURED rather than assumed: the block rendered around a
    one-byte payload, minus that byte.

    A2 owns the joins — it drops an empty payload and an empty anchor block entirely — so counting
    them here would be a second, drifting copy of the grammar.
    """
    rendered = render_summary_block(boundary_ts=boundary_ts, payload="x", anchor_block="",
                                    stale=stale)
    return admission_charge(rendered) - 1


def anchor_block_bytes(*, boundary_ts: str, anchor_block: str, stale: bool = False) -> int:
    """What the anchor block ADDS to the rendered summary item, its join included."""
    with_anchors = render_summary_block(boundary_ts=boundary_ts, payload="x",
                                        anchor_block=anchor_block, stale=stale)
    return admission_charge(with_anchors) - 1 - summary_shell_bytes(boundary_ts=boundary_ts,
                                                                    stale=stale)


def summary_byte_cap(*, room_tokens: int, configured_cap: Optional[int] = None) -> int:
    """`cap = min(SUMMARY_BYTE_CAP, available_room_tokens)`."""
    ceiling = int(config.summary_byte_cap if configured_cap is None
                  else configured_cap)
    return max(0, min(ceiling, int(room_tokens)))


def validate_summary(text: Any, *, cap: int) -> Optional[str]:
    """None when the output is publishable, else the reason it is not.

    An over-cap output is REJECTED AS MALFORMED and takes the ordinary discard-and-retry path;
    a "~2000-token target" is an instruction to the model, and this is the validation.
    """
    body = str(text or "")
    if not body.strip():
        return "empty_output"
    if admission_charge(body) > int(cap):
        return "over_byte_cap"
    return None


# ---------------------------------------------------------------- receipts and membership

def freeze_receipts(receipts: Iterable[Mapping[str, Any]], *,
                    epoch_ts: Optional[str],
                    chrome_ts: Sequence[str] = (),
                    self_ts: Sequence[str] = ()) -> Dict[str, str]:
    """The §1e PROOF STATES for our own messages, frozen at the source pin.

    `in_flight` is kept deliberately: it is valid evidence for the BOUNDARY CAP and remains
    invalid as proof of summarizable inclusion. Omitting it would leave boundary selection unable
    to compute its own cap.
    """
    states: Dict[str, str] = {}
    rows = {str(row.get("message_ts") or row.get("ts")): row for row in receipts or ()}
    chrome = {str(t) for t in chrome_ts}
    for ts in {str(t) for t in self_ts} | set(rows) | chrome:
        row = rows.get(ts)
        if ts in chrome:
            states[ts] = PROOF_CHROME
            continue
        if row is not None:
            state = str(row.get("state") or "")
            states[ts] = PROOF_FINALIZED if state == "finalized" else PROOF_IN_FLIGHT
            continue
        if epoch_ts:
            try:
                pre_epoch = parse_ts(ts) < parse_ts(epoch_ts)
            except TimestampError:
                pre_epoch = False
            if pre_epoch:
                states[ts] = PROOF_GRANDFATHERED
                continue
        states[ts] = PROOF_UNPROVEN
    return states


def receipt_proof(ts: str, frozen_receipts: Mapping[str, str]) -> str:
    return str(frozen_receipts.get(str(ts)) or PROOF_UNPROVEN)


def is_skeleton_member(message: NormalizedMessage, *,
                       frozen_receipts: Mapping[str, str]) -> bool:
    """MEMBERSHIP: chrome, receipt-excluded own messages and anything the normalizer filters are
    ABSENT from the sealed skeleton entirely.

    Giving them rows would move 500-event chunk edges — and therefore `source_hash` — without
    moving a single projection byte.
    """
    if message.sender_type != "self":
        return True
    return receipt_proof(message.ts, frozen_receipts) in SUMMARIZABLE_PROOFS


def receipt_constrained_cap(frozen_receipts: Mapping[str, str]) -> Optional[str]:
    """The OLDEST `in_flight` own-message ts. The boundary must sit STRICTLY BELOW it.

    Computed FIRST, before any sizing: an old in-flight receipt can force the legal boundary far
    earlier than size alone would, and discovering it afterwards would mean discarding the
    candidate and starting over.
    """
    oldest: Optional[str] = None
    for ts, state in (frozen_receipts or {}).items():
        if state != PROOF_IN_FLIGHT:
            continue
        try:
            if oldest is None or parse_ts(ts) < parse_ts(oldest):
                oldest = str(ts)
        except TimestampError:
            continue
    return oldest


# ---------------------------------------------------------------- boundary selection

@dataclass(frozen=True)
class BoundaryCandidate:
    boundary_ts: str
    index: int
    retained_charge: int
    retained_items: int
    retained_messages: int


def _skeleton_field(row: Mapping[str, Any], name: str, default: Any = 0) -> Any:
    value = row.get(name)
    return default if value is None else value


def retained_charge(rows: Sequence[Mapping[str, Any]], *, index: int,
                    root_snippet_lens: Mapping[str, int],
                    item_overhead: int = ITEM_STRUCTURAL_OVERHEAD) -> int:
    """The CANDIDATE FORMULA (§1e), evaluated directly.

        charge(B) = cumulative base charge of retained events
                  + root_snippet_len for each retained REPLY whose root ALSO survives B
                  + retained_item_count × ITEM_STRUCTURAL_OVERHEAD

    Both extra terms move with B, which is why `cum_base_charge` is stored boundary-independent
    and is only a LOWER BOUND. Deciding fit from that aggregate alone is wrong.
    """
    boundary_ts = rows[index]["ts"] if 0 <= index < len(rows) else None
    total = 0
    count = 0
    for row in rows[index + 1:]:
        count += 1
        total += int(_skeleton_field(row, "base_canonical_bytes"))
        root = str(row.get("root_ts") or ROOT_SENTINEL)
        if root == ROOT_SENTINEL or int(_skeleton_field(row, "kind_rank")) == KIND_RANK_MESSAGE:
            continue
        if boundary_ts is not None and parse_ts(root) <= parse_ts(boundary_ts):
            continue  # the root falls pre-boundary; it is represented by an anchor instead
        total += int(root_snippet_lens.get(root, 0))
    return total + count * int(item_overhead)


def starting_index(chunk_aggregates: Sequence[Mapping[str, Any]], *, total_base: int,
                   budget_tokens: int) -> int:
    """Walk chunk aggregates by `cum_base_charge` to reach a STARTING chunk.

    Boundary-independent and only a LOWER BOUND, so it narrows the search and never decides fit.
    """
    start = 0
    for entry in chunk_aggregates or ():
        if int(_skeleton_field(entry, "events")) <= 0:
            continue  # chunk zero of an incremental crawl; boundary search skips it
        remaining = total_base - int(_skeleton_field(entry, "cum_base_charge"))
        if remaining <= budget_tokens:
            break
        start = int(_skeleton_field(entry, "seq_end", start))
    return max(0, start)


def select_boundary(*, rows: Sequence[Mapping[str, Any]],
                    root_snippet_lens: Mapping[str, int],
                    budget_tokens: int,
                    min_tail: Optional[int] = None,
                    cap_ts: Optional[str] = None,
                    high_ts: Optional[str] = None,
                    prior_boundary_ts: Optional[str] = None,
                    chunk_aggregates: Sequence[Mapping[str, Any]] = (),
                    from_index: int = 0,
                    item_overhead: int = ITEM_STRUCTURAL_OVERHEAD
                    ) -> Optional[BoundaryCandidate]:
    """The first boundary that FITS, or None.

    Every sealed row is a model-visible message event, so the boundary always lands exactly on a
    canonical message and `COMPACTION_MIN_TAIL` is counted in rows. Sizing walks the skeleton and
    issues no Slack call at all. Fit is proven in the CANONICAL charge, never the projection
    charge.

    The scan CROSSES CHUNK EDGES: the snippet and overhead terms can push the first fitting
    boundary past the chunk the lower bound stopped at, and a search that gave up at that edge
    would report "no fit" where one exists.
    """
    total = len(rows)
    if total == 0:
        return None
    tail = int(config.compaction_min_tail if min_tail is None else min_tail)
    highest = total - 1 - max(0, tail)
    if highest < 0:
        return None

    total_base = sum(int(_skeleton_field(r, "base_canonical_bytes")) for r in rows)
    lower = max(int(from_index), starting_index(chunk_aggregates, total_base=total_base,
                                                budget_tokens=budget_tokens))
    lower = min(lower, highest)

    suffix_base = [0] * (total + 1)
    for i in range(total - 1, -1, -1):
        suffix_base[i] = suffix_base[i + 1] + int(_skeleton_field(rows[i],
                                                                  "base_canonical_bytes"))

    # Snippet term, maintained incrementally so the forward scan stays linear.
    retained_replies: Dict[str, int] = {}
    for row in rows[lower + 1:]:
        if int(_skeleton_field(row, "kind_rank")) == KIND_RANK_MESSAGE:
            continue
        root = str(row.get("root_ts") or ROOT_SENTINEL)
        if root != ROOT_SENTINEL:
            retained_replies[root] = retained_replies.get(root, 0) + 1

    def _root_ts_key(value: str) -> Tuple[int, int]:
        return parse_ts(value)

    surviving = sorted(retained_replies, key=_root_ts_key)
    alive: Set[str] = set()
    snippet_total = 0
    boundary_ts = str(rows[lower]["ts"]) if lower >= 0 else None
    pointer = 0
    for root in surviving:
        if boundary_ts is not None and _root_ts_key(root) <= parse_ts(boundary_ts):
            pointer += 1
            continue
        alive.add(root)
        snippet_total += int(root_snippet_lens.get(root, 0)) * retained_replies[root]

    def _charge(index: int) -> int:
        count = total - index - 1
        return suffix_base[index + 1] + snippet_total + count * int(item_overhead)

    def _legal(index: int) -> bool:
        ts = str(rows[index]["ts"])
        if prior_boundary_ts and parse_ts(ts) <= parse_ts(prior_boundary_ts):
            return False
        if high_ts and parse_ts(ts) > parse_ts(high_ts):
            return False
        if cap_ts and parse_ts(ts) >= parse_ts(cap_ts):
            return False
        return True

    index = lower
    while index <= highest:
        if _legal(index) and _charge(index) <= budget_tokens:
            return BoundaryCandidate(
                boundary_ts=str(rows[index]["ts"]), index=index,
                retained_charge=_charge(index), retained_items=total - index - 1,
                retained_messages=total - index - 1)
        # step the boundary up one row, maintaining the snippet term
        index += 1
        if index > highest:
            break
        leaving = rows[index]
        if int(_skeleton_field(leaving, "kind_rank")) != KIND_RANK_MESSAGE:
            root = str(leaving.get("root_ts") or ROOT_SENTINEL)
            if root != ROOT_SENTINEL:
                retained_replies[root] = retained_replies.get(root, 1) - 1
                if root in alive:
                    snippet_total -= int(root_snippet_lens.get(root, 0))
        boundary_ts = str(rows[index]["ts"])
        while pointer < len(surviving) and _root_ts_key(surviving[pointer]) <= parse_ts(
                boundary_ts):
            root = surviving[pointer]
            if root in alive:
                alive.discard(root)
                snippet_total -= int(root_snippet_lens.get(root, 0)) * max(
                    0, retained_replies.get(root, 0))
            pointer += 1
    return None


def fit_result(*, total_charge: int, target_tokens: int, trigger_tokens: int) -> str:
    """The §1e postcondition, remeasured AFTER generation."""
    if total_charge <= target_tokens:
        return FIT_UNDER_TARGET
    if total_charge <= trigger_tokens:
        return FIT_UNDER_TRIGGER
    return FIT_NONE


# ---------------------------------------------------------------- telemetry accumulators

@dataclass
class CompactionAttempt:
    """ONE generation attempt, and the accumulators §1l aggregates into its single `op=build`.

    An attempt spans every resumption until it publishes, fails or is discarded, so the totals
    live on the checkpoint and are reloaded rather than recounted.
    """
    crawl_id: str
    attempt_seq: int
    team_id: str
    channel_id: str
    namespace: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached_input_tokens: int = 0
    call_count: int = 0
    charges: List[int] = field(default_factory=list)

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any], *, model: str) -> "CompactionAttempt":
        return cls(
            crawl_id=str(checkpoint.get("crawl_id") or ""),
            attempt_seq=int(checkpoint.get("attempt_seq") or 0),
            team_id=str(checkpoint.get("team_id") or ""),
            channel_id=str(checkpoint.get("channel_id") or ""),
            namespace=str(checkpoint.get("namespace") or ""),
            model=model,
            tokens_in=int(checkpoint.get("attempt_tokens_in") or 0),
            tokens_out=int(checkpoint.get("attempt_tokens_out") or 0),
            cached_input_tokens=int(checkpoint.get("attempt_cached_input_tokens") or 0),
            call_count=int(checkpoint.get("attempt_call_count") or 0),
        )

    def charge(self, text: str) -> int:
        """The compaction request charged with `admission_charge` BEFORE its API call — every
        map call and every level of the hierarchical reduce."""
        cost = admission_charge(text)
        self.charges.append(cost)
        return cost

    def record_call(self, usage: Optional[Mapping[str, Any]] = None) -> None:
        self.call_count += 1
        usage = usage or {}
        self.tokens_in += int(usage.get("input_tokens") or 0)
        self.tokens_out += int(usage.get("output_tokens") or 0)
        self.cached_input_tokens += int(usage.get("cached_input_tokens") or 0)

    def checkpoint_patch(self) -> Dict[str, Any]:
        return {
            "attempt_seq": self.attempt_seq,
            "attempt_tokens_in": self.tokens_in,
            "attempt_tokens_out": self.tokens_out,
            "attempt_cached_input_tokens": self.cached_input_tokens,
            "attempt_call_count": self.call_count,
        }

    def _identity(self, event_seq: int, at: float) -> Dict[str, Any]:
        return {
            "event": "compaction_snapshot",
            "crawl_id": self.crawl_id,
            "attempt_seq": self.attempt_seq,
            "event_seq": event_seq,
            "team_id": self.team_id,
            "channel_id": self.channel_id,
            "namespace": self.namespace,
            "at": float(at),
        }

    def build_body(self, *, status: str, at: float,
                   reason: Optional[str] = None) -> Dict[str, Any]:
        """`op=build` at `event_seq = 0`. The authoritative token names, no aliases."""
        body = self._identity(0, at)
        body.update({
            "op": "build",
            "model": self.model,
            "tokens_in": int(self.tokens_in),
            "tokens_out": int(self.tokens_out),
            "cached_input_tokens": int(self.cached_input_tokens),
            "call_count": int(self.call_count),
            "status": status,
        })
        if status in (BUILD_FAILED, BUILD_DISCARDED):
            body["reason"] = str(reason or status)
        return body

    def publish_body(self, *, at: float, snapshot_id: str, generation: int, boundary_ts: str,
                     fit: str, serializer_version: int) -> Dict[str, Any]:
        """`op=publish` at `event_seq = 1`. NO token fields — the cost belongs to the build."""
        body = self._identity(1, at)
        body.update({
            "op": "publish",
            "snapshot_id": snapshot_id,
            "generation": int(generation),
            "boundary_ts": str(boundary_ts),
            "fit_result": fit,
            "serializer_version": int(serializer_version),
        })
        return body

    def outbox_row(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        return {"crawl_id": self.crawl_id, "attempt_seq": self.attempt_seq,
                "event_seq": int(body["event_seq"]), "body": dict(body)}


# ---------------------------------------------------------------- generation

async def _one_call(openai_client: Any, *, attempt: CompactionAttempt, prompt: str, body: str,
                    max_output_tokens: int, cache_key: Optional[str]) -> str:
    """One Responses API call: `store=False`, NO tools, utility model, `ANALYSIS_*`.

    Temperature is passed through unchanged — the landed wrapper's behaviour is deliberately not
    changed here (Appendix B3).
    """
    attempt.charge(prompt + "\n" + body)
    usage: Dict[str, Any] = {}
    text = await openai_client.create_text_response(
        messages=[{"role": "developer", "content": prompt},
                  {"role": "user", "content": body}],
        model=attempt.model,
        max_tokens=int(max_output_tokens),
        reasoning_effort=getattr(config, "analysis_reasoning_effort", None),
        verbosity=getattr(config, "analysis_verbosity", None),
        store=False,
        system_prompt=None,
        prompt_cache_key=cache_key,
        usage_sink=usage,
    )
    attempt.record_call(usage)
    return str(text or "")


async def run_map_stage(openai_client: Any, *, attempt: CompactionAttempt,
                        chunks: Sequence[Tuple[Tuple[int, int], Sequence[str]]],
                        max_output_tokens: int,
                        cache_key: Optional[str] = None) -> Dict[str, str]:
    """Summarize each MAP chunk. Keys are `(hash_chunk_idx, sub_idx)`."""
    out: Dict[str, str] = {}
    for (chunk_idx, sub_idx), lines in chunks:
        text = await _one_call(openai_client, attempt=attempt,
                               prompt=COMPACTION_MAP_PROMPT_V1,
                               body="\n".join(lines) + "\n",
                               max_output_tokens=max_output_tokens, cache_key=cache_key)
        out[f"{chunk_idx}:{sub_idx}"] = text.strip()
    return out


async def hierarchical_reduce(openai_client: Any, *, attempt: CompactionAttempt,
                              summaries: Sequence[str], bound: int, budget_tokens: int,
                              max_output_tokens: int,
                              cache_key: Optional[str] = None) -> str:
    """The §1g HIERARCHICAL REDUCE.

    Map chunks are bounded but the number of map RESULTS is not, so a single reduce call can
    overflow. Group greedily in key order, reduce each group, repeat until one summary remains.
    EVERY level's calls are charged and aggregated into the single `op=build`.
    """
    level = [s for s in summaries if str(s).strip()]
    if not level:
        return ""
    prompt = COMPACTION_REDUCE_PROMPT_V1.format(budget=int(budget_tokens))
    base = admission_charge(prompt) + 1
    while len(level) > 1:
        groups = reduce_groups(level, bound=bound, base_charge=base)
        if len(groups) >= len(level):
            # Every group holds one summary, so another level would change nothing and spin.
            # One over-bound call is the honest way out.
            logger.warning(
                "compaction reduce cannot group further; issuing one over-bound reduce call")
            groups = [list(level)]
        produced: List[str] = []
        for group in groups:
            text = await _one_call(openai_client, attempt=attempt, prompt=prompt,
                                   body="\n\n".join(group) + "\n",
                                   max_output_tokens=max_output_tokens, cache_key=cache_key)
            produced.append(text.strip())
        level = [t for t in produced if t] or produced
    # Exactly one summary remains; that is the payload input. A single map chunk needs no reduce.
    return level[0] if level else ""


# ---------------------------------------------------------------- anchors

@dataclass(frozen=True)
class AnchorEntry:
    root_ts: str
    status: str
    author_id: Optional[str]
    text: Optional[str]
    tombstone: bool
    receipt_proof_state: Optional[str]
    observation_frontier: int
    projection_sha256: str

    def render_entry(self) -> Dict[str, Any]:
        return {"root_ts": self.root_ts, "author_id": self.author_id, "text": self.text,
                "status": self.status, "tombstone": self.tombstone}

    def provenance_row(self, *, team_id: str) -> Dict[str, Any]:
        return {"team_id": team_id, "root_ts": self.root_ts, "status": self.status,
                "projection_sha256": self.projection_sha256,
                "observation_frontier": int(self.observation_frontier),
                "receipt_proof": self.receipt_proof_state}


def straddling_roots(root_inventory: Mapping[str, Mapping[str, Any]], *, boundary_ts: str,
                     bound: Optional[int] = None) -> Tuple[List[str], int]:
    """The anchored set: `root_ts <= boundary_ts < last_canonical_message_ts`.

    Only STRADDLING threads are anchored at all — a thread with nothing in the rendered window
    needs no referent, and anchoring it would spend the map bound on a thread the model cannot
    see. A TOMBSTONE counts as straddle evidence: `last_canonical_message_ts` is advanced by any
    canonical, model-visible event, and the normalizer keeps tombstones.
    """
    limit = int(config.snapshot_anchor_map_bound if bound is None else bound)
    eligible: List[str] = []
    for root_ts, entry in (root_inventory or {}).items():
        last = entry.get("last_canonical_message_ts")
        if not last:
            continue
        try:
            if parse_ts(root_ts) > parse_ts(boundary_ts):
                continue
            if parse_ts(last) <= parse_ts(boundary_ts):
                continue
        except TimestampError:
            continue
        eligible.append(str(root_ts))
    eligible.sort(key=parse_ts)
    if limit >= 0 and len(eligible) > limit:
        return eligible[:limit], len(eligible) - limit
    return eligible, 0


def _anchor_fingerprint(entry: Mapping[str, Any]) -> str:
    """DURABLE RENDERED-BYTE PROVENANCE — never a publication-time comparison. A publication-time
    Slack refetch is forbidden and none is added."""
    return hashlib.sha256(render_anchor_block([entry], omitted=0).encode("utf-8")).hexdigest()


def classify_anchor_root(message: Optional[NormalizedMessage], *,
                         frozen_receipts: Mapping[str, str],
                         chrome_ts: Sequence[str] = ()) -> Tuple[str, Optional[str]]:
    """The SAME §1e receipt/chrome classification the serializer applies to own messages.

    The normalizer alone does not make that decision, and a pre-floor root of ours could
    otherwise expose status chrome or an unsafe post-epoch own message as anchor text.
    """
    if message is None:
        return ANCHOR_UNAVAILABLE, None
    if message.sender_type != "self":
        return ANCHOR_AVAILABLE, RECEIPT_PROOF_NOT_SELF
    if str(message.ts) in {str(t) for t in chrome_ts}:
        return ANCHOR_UNSAFE, None
    proof = receipt_proof(message.ts, frozen_receipts)
    if proof in SUMMARIZABLE_PROOFS:
        return ANCHOR_AVAILABLE, proof
    return ANCHOR_UNSAFE, None


async def resolve_anchors(*, db: Any, client: Any, team_id: str, channel_id: str,
                          roots: Sequence[str], frozen_receipts: Mapping[str, str],
                          chrome_ts: Sequence[str] = (),
                          anchor_text_chars: int = 240) -> List[AnchorEntry]:
    """ONE BOUNDED TARGETED REFETCH of each anchored root's SINGLE root message.

    Transaction order is pinned: capture each root's `observation_frontier` BEFORE that root's
    fetch — capturing it afterwards would BLESS a mutation that landed between fetch and read —
    then fetch, classify, fingerprint. The rows are inserted WITH THE CANDIDATE.

    Pre-floor roots are fetched DIRECTLY BY TS; a root older than the input floor is exactly the
    case a history walk cannot reach.
    """
    method = _web(client, "conversations_history")
    entries: List[AnchorEntry] = []
    for root_ts in roots:
        frontier = await _max_observation_id(db, team_id, channel_id)
        status = ANCHOR_UNAVAILABLE
        message: Optional[NormalizedMessage] = None
        if method is None:
            status = ANCHOR_REFUSED
        else:
            try:
                page = await fetch_page(method, {
                    "channel": channel_id, "latest": root_ts, "oldest": root_ts,
                    "inclusive": True, "limit": 1}, label="anchor root")
                raw = next((m for m in page.messages
                            if str(m.get("ts")) == str(root_ts)), None)
                if raw is not None:
                    message = normalize_slack_message(
                        client, raw, channel_id=channel_id, origin=ORIGIN_HISTORY,
                        team_id=team_id, allow_any_subtype=True)
            except HistoryFetchError as e:
                logger.warning(f"anchor root {channel_id}/{root_ts} not fetched: {e}")
                status = ANCHOR_REFUSED
            except TimestampError as e:
                logger.warning(f"anchor root {channel_id}/{root_ts} unusable: {e}")
                status = ANCHOR_REFUSED
        proof: Optional[str] = None
        if status == ANCHOR_UNAVAILABLE:
            status, proof = classify_anchor_root(message, frozen_receipts=frozen_receipts,
                                                 chrome_ts=chrome_ts)
        tombstone = bool(message is not None and message.is_tombstone)
        text = None
        author = None
        if status == ANCHOR_AVAILABLE and message is not None:
            author = message.sender_id or "unknown"
            # Truncation and the stricter quoted-field escaping are A2's (`escape_anchor_text`),
            # applied at render time on the SOURCE text; pre-escaping here would double-escape.
            text = message.text
        entry = AnchorEntry(root_ts=str(root_ts), status=status, author_id=author, text=text,
                            tombstone=tombstone, receipt_proof_state=proof,
                            observation_frontier=int(frontier), projection_sha256="")
        entry = AnchorEntry(**{**entry.__dict__,
                               "projection_sha256": _anchor_fingerprint(entry.render_entry())})
        entries.append(entry)
    return entries


# ---------------------------------------------------------------- crawl plumbing

def _web(client: Any, name: str) -> Optional[Callable[..., Any]]:
    app = getattr(client, "app", None)
    web = getattr(app, "client", None) if app is not None else None
    method = getattr(web, name, None)
    if callable(method):
        return method
    method = getattr(client, name, None)
    return method if callable(method) else None


async def _max_observation_id(db: Any, team_id: str, channel_id: str) -> int:
    getter = getattr(db, "max_mutation_observation_id_async", None)
    if not callable(getter):
        return 0
    return int(await getter(team_id, channel_id) or 0)


def load_json_field(value: Any, default: Any) -> Any:
    """A checkpoint JSON column, whether the accessor handed back TEXT or a decoded object."""
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def dump_json_field(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def actor_snapshot_hash(actor_map: Mapping[str, str]) -> str:
    return hashlib.sha256(dump_json_field(dict(actor_map)).encode("utf-8")).hexdigest()


class SliceBudget:
    """CRAWL_PAGE_BUDGET / CRAWL_TIME_BUDGET, per WORKER SLICE — never per attempt.

    Phase I checks it per page; phase II checks it BETWEEN chunks. A started chunk always runs to
    completion, so exhausting the budget defers the next chunk and is not a failure.
    """

    def __init__(self, *, pages: Optional[int] = None, seconds: Optional[float] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._started = clock()
        self.page_budget = int(getattr(config, "crawl_page_budget", 500)
                               if pages is None else pages)
        self.time_budget = float(getattr(config, "crawl_time_budget", 600.0)
                                 if seconds is None else seconds)
        self.pages_used = 0

    def charge_page(self) -> None:
        self.pages_used += 1

    def remaining_pages(self) -> int:
        return self.page_budget - self.pages_used

    def exhausted(self) -> bool:
        return (self.pages_used >= self.page_budget
                or (self._clock() - self._started) >= self.time_budget)


@dataclass
class SkeletonRow:
    ts: str
    root_ts: str
    kind_rank: int
    source_rank: int
    actor_id: str
    projected_byte_len: int
    base_canonical_bytes: int
    projection_sha256: str

    def as_row(self) -> Dict[str, Any]:
        return {"seq": None, "ts": self.ts, "root_ts": self.root_ts,
                "kind_rank": self.kind_rank, "source_rank": self.source_rank,
                "actor_id": self.actor_id, "projected_byte_len": self.projected_byte_len,
                "base_canonical_bytes": self.base_canonical_bytes,
                "projection_sha256": self.projection_sha256}


def base_canonical_bytes(message: NormalizedMessage, *,
                         actor_names: Mapping[str, str],
                         marker_lines: Sequence[str] = ()) -> int:
    """The event's CANONICAL content length, bundled markers INCLUDED, root snippet and
    per-item structural overhead EXCLUDED.

    Bundling is admission accounting, not only projection accounting: the retained stream renders
    attachment and tool-provenance markers as canonical bytes too, so counting them for
    `projected_byte_len` and not here would undercharge every retained event carrying an artifact.
    """
    from message_processor import channel_stream as cs

    root = None  # the snippet is composed per candidate boundary, never folded in here
    lines = [cs.render_header(message, actor_names=dict(actor_names), root=root)]
    lines.extend(cs.render_body(message, dict(actor_names)))
    files_marker = cs.render_files_marker(message)
    if files_marker:
        lines.append(files_marker)
    lines.extend(marker_lines)
    reactions_marker = cs.render_reactions_marker(message)
    if reactions_marker:
        lines.append(reactions_marker)
    return admission_charge("\n".join(lines))


def build_skeleton_row(message: NormalizedMessage, *, actor_names: Mapping[str, str],
                       artifact_renders: Sequence[Mapping[str, Any]] = (),
                       marker_lines: Sequence[str] = (),
                       source_rank: int = SOURCE_RANK_HISTORY) -> SkeletonRow:
    projection = project_event(message, actor_names=actor_names,
                               artifact_renders=artifact_renders)
    return SkeletonRow(
        ts=str(message.ts), root_ts=root_key(message), kind_rank=kind_rank(message),
        source_rank=source_rank, actor_id=str(message.sender_id or "unknown"),
        projected_byte_len=admission_charge(projection),
        base_canonical_bytes=base_canonical_bytes(message, actor_names=actor_names,
                                                  marker_lines=marker_lines),
        projection_sha256=event_fingerprint(projection))


def merge_events(batches: Sequence[Sequence[NormalizedMessage]]) -> List[NormalizedMessage]:
    """k-way merge IN MEMORY, using the canonical normalizer's dedup unchanged: one record per
    ts, the REPLIES copy winning, sorted by the composite triple."""
    chosen: Dict[str, NormalizedMessage] = {}
    for batch in batches:
        for message in batch:
            current = chosen.get(message.ts)
            if current is None or (current.origin != ORIGIN_REPLIES
                                   and message.origin == ORIGIN_REPLIES):
                chosen[message.ts] = message
    return sorted(chosen.values(), key=order_key)


def verify_slice(events: Sequence[NormalizedMessage], rows: Sequence[Mapping[str, Any]], *,
                 actor_names: Mapping[str, str],
                 renders_for: Optional[Callable[[str], Sequence[Mapping[str, Any]]]] = None
                 ) -> List[str]:
    """MATCH by `(ts, kind_rank)`, then VERIFY `projection_sha256`.

    An unmatched event, an unmatched skeleton row, or a fingerprint mismatch is a MUTATION and
    takes the DISCARD path — the fingerprint is what catches a SAME-LENGTH edit that changes
    neither the identity nor either byte count.
    """
    by_identity = {(str(r["ts"]), int(r["kind_rank"])): r for r in rows}
    projections: List[str] = []
    seen: Set[Tuple[str, int]] = set()
    for message in events:
        key = identity_key(message)
        row = by_identity.get(key)
        if row is None:
            raise SourceMutated(f"event {key} has no skeleton row", subject_ts=str(message.ts))
        seen.add(key)
        renders = renders_for(str(message.ts)) if renders_for else ()
        projection = project_event(message, actor_names=actor_names,
                                   artifact_renders=renders)
        if event_fingerprint(projection) != str(row.get("projection_sha256")):
            raise SourceMutated(f"fingerprint mismatch at {message.ts}",
                                subject_ts=str(message.ts))
        projections.append(projection)
    missing = set(by_identity) - seen
    if missing:
        ts = sorted(missing)[0][0]
        raise SourceMutated(f"skeleton row {sorted(missing)[0]} has no event", subject_ts=ts)
    return projections


def chunk_refetch_plan(rows: Sequence[Mapping[str, Any]], *,
                       root_inventory: Mapping[str, Mapping[str, Any]],
                       history_span_density: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The chunk's refetch list FROM THE SKELETON SLICE, with its cost estimated from phase I's
    OBSERVED RAW PAGE COUNTS.

    Slack pages carry chrome, duplicate broadcasts and filtered records that never reach the
    skeleton, so the skeleton's event counts are a LOWER BOUND on page cost, not the cost.
    """
    roots = sorted({str(r.get("root_ts")) for r in rows
                    if str(r.get("root_ts") or ROOT_SENTINEL) != ROOT_SENTINEL}, key=parse_ts)
    span_start = min((str(r["ts"]) for r in rows), key=parse_ts, default=None)
    span_end = max((str(r["ts"]) for r in rows), key=parse_ts, default=None)
    pages = 0
    for root in roots:
        pages += int((root_inventory.get(root) or {}).get("observed_raw_pages") or 1)
    for span in history_span_density or ():
        start, end = span.get("span_start_ts"), span.get("span_end_ts")
        if not (start and end and span_start and span_end):
            continue
        if parse_ts(end) < parse_ts(span_start) or parse_ts(start) > parse_ts(span_end):
            continue
        pages += int(span.get("observed_raw_pages") or 1)
    return {"roots": roots, "span_start_ts": span_start, "span_end_ts": span_end,
            "estimated_pages": max(pages, 1)}


# ---------------------------------------------------------------- resume / discard

def reset_reason(checkpoint: Mapping[str, Any], live: Mapping[str, Any]) -> Optional[str]:
    """The step-2 config/version comparison. Any difference discards progress."""
    for name in RESET_FIELDS:
        if name not in live:
            continue
        if str(checkpoint.get(name)) != str(live.get(name)):
            return name
    return None


def verify_actor_snapshot(checkpoint: Mapping[str, Any]) -> bool:
    """`actor_snapshot_hash` is INTEGRITY VERIFICATION ONLY — recomputed over the PERSISTED map
    and compared to the stored hash.

    It is never compared against the live resolver: that would be re-resolution by another name
    and would reset a healthy crawl every time somebody changed their display name.
    """
    stored = str(checkpoint.get("actor_snapshot_hash") or "")
    if not stored:
        return True
    actors = load_json_field(checkpoint.get("actor_snapshot"), {})
    return actor_snapshot_hash(actors) == stored


def apply_config_reset(checkpoint: Mapping[str, Any], *, live: Mapping[str, Any],
                       mutation_frontier: int,
                       crawl_id: Optional[str] = None) -> Dict[str, Any]:
    """A step-2 reset does NOT increment `consecutive_discards` and PRESERVES an active
    `next_attempt_after` — a config change cannot buy an escape from backoff."""
    patch = dict(checkpoint)
    patch.update(live)
    patch.update({
        "crawl_id": crawl_id or uuid.uuid4().hex,
        "phase": PHASE_INVENTORY,
        "chunk_index": 0,
        "chunk_hashes": dump_json_field([]),
        "chunk_aggregates": dump_json_field([]),
        "chunk_summaries": dump_json_field({}),
        "root_inventory": dump_json_field({}),
        "history_span_density": dump_json_field([]),
        "actor_snapshot": dump_json_field({}),
        "actor_snapshot_hash": "",
        "inventory_cursor_ts": None,
        "boundary_ts": None,
        "event_count": 0,
        "mutation_frontier": int(mutation_frontier),
        "attempt_seq": int(checkpoint.get("attempt_seq") or 0) + 1,
        "attempt_tokens_in": 0, "attempt_tokens_out": 0,
        "attempt_cached_input_tokens": 0, "attempt_call_count": 0,
        # preserved deliberately
        "consecutive_discards": int(checkpoint.get("consecutive_discards") or 0),
        "next_attempt_after": checkpoint.get("next_attempt_after"),
    })
    return patch


def apply_mutation_discard(checkpoint: Mapping[str, Any], *, mutation_frontier: int,
                           crawl_id: Optional[str] = None,
                           now: Optional[float] = None) -> Dict[str, Any]:
    """The step-3 MUTATION-driven discard — the ONLY one that increments
    `consecutive_discards`."""
    count = int(checkpoint.get("consecutive_discards") or 0) + 1
    patch = dict(checkpoint)
    patch.update({
        "crawl_id": crawl_id or uuid.uuid4().hex,
        "phase": PHASE_INVENTORY,
        "chunk_index": 0,
        "chunk_hashes": dump_json_field([]),
        "chunk_aggregates": dump_json_field([]),
        "chunk_summaries": dump_json_field({}),
        "root_inventory": dump_json_field({}),
        "history_span_density": dump_json_field([]),
        "actor_snapshot": dump_json_field({}),
        "actor_snapshot_hash": "",
        "inventory_cursor_ts": None,
        "boundary_ts": None,
        "event_count": 0,
        "mutation_frontier": int(mutation_frontier),
        "consecutive_discards": count,
        "attempt_seq": int(checkpoint.get("attempt_seq") or 0) + 1,
        "attempt_tokens_in": 0, "attempt_tokens_out": 0,
        "attempt_cached_input_tokens": 0, "attempt_call_count": 0,
    })
    deadline = stall_deadline(count, now=now)
    if deadline is not None:
        patch["next_attempt_after"] = deadline
    return patch


def stall_deadline(consecutive_discards: int, *,
                   now: Optional[float] = None) -> Optional[str]:
    """The stall guard fires on EVERY discard at count >= 3, with a FRESH 1-hour deadline. A
    single delay would let the fourth discard, arriving after it expired, spin freely."""
    if int(consecutive_discards) < STALL_DISCARD_THRESHOLD:
        return None
    return str((time.time() if now is None else float(now)) + STALL_BACKOFF_SECONDS)


def backoff_active(checkpoint: Mapping[str, Any], *, now: Optional[float] = None) -> bool:
    raw = checkpoint.get("next_attempt_after")
    if not raw:
        return False
    try:
        return float(raw) > (time.time() if now is None else float(now))
    except (TypeError, ValueError):
        return False


async def mutation_discard_needed(db: Any, checkpoint: Mapping[str, Any]
                                  ) -> Optional[Mapping[str, Any]]:
    """Step 3: any observation with `id > mutation_frontier` whose `subject_ts` falls in
    `[source_floor_ts, pinned_H]` — the whole span the crawl is summarizing.

    The predicate is deliberately WIDE: during phase I the boundary is not yet known, so a
    narrower one cannot be shown sound.
    """
    getter = getattr(db, "mutation_observations_after_async", None)
    if not callable(getter):
        return None
    rows = await getter(str(checkpoint.get("team_id")), str(checkpoint.get("channel_id")),
                        int(checkpoint.get("mutation_frontier") or 0),
                        floor_ts=str(checkpoint.get("source_floor_ts")),
                        high_ts=str(checkpoint.get("pinned_H")))
    return rows[0] if rows else None


# ---------------------------------------------------------------- phase I

async def _history_page(client: Any, *, channel_id: str, latest: str, inclusive: bool,
                        limit: int) -> PageResult:
    method = _web(client, "conversations_history")
    if method is None:
        raise HistoryFetchError("no conversations.history method on this client")
    # `oldest` is deliberately omitted: Slack's single `inclusive` flag applies to BOTH bounds,
    # so pinning it False for the backward cursor would also drop a message sitting exactly on an
    # inclusive input floor. The floor is applied locally by the window predicate instead, and the
    # walk stops as soon as a page reaches it.
    return await fetch_page(method, {"channel": channel_id, "latest": latest,
                                     "inclusive": bool(inclusive), "limit": int(limit)},
                            label="crawl history")


async def _replies_page(client: Any, *, channel_id: str, root_ts: str,
                        oldest: Optional[str], high: str, limit: int) -> PageResult:
    method = _web(client, "conversations_replies")
    if method is None:
        raise HistoryFetchError("no conversations.replies method on this client")
    params: Dict[str, Any] = {"channel": channel_id, "ts": root_ts, "latest": high,
                              "inclusive": True, "limit": int(limit)}
    if oldest:
        # The cursor is an EXCLUSIVE lower bound, enforced locally for the same reason the
        # history walk omits `oldest`: one flag cannot mean two things.
        params["oldest"] = str(oldest)
    return await fetch_page(method, params, label="crawl replies")


def _page_messages(client: Any, raw: Sequence[Dict[str, Any]], *, channel_id: str,
                   team_id: str, origin: str) -> List[NormalizedMessage]:
    out: List[NormalizedMessage] = []
    for payload in raw:
        message = normalize_slack_message(client, payload, channel_id=channel_id,
                                          origin=origin, team_id=team_id)
        if message is not None:
            out.append(message)
    return out


async def _resolve_new_actors(client: Any, messages: Sequence[NormalizedMessage],
                              actors: Dict[str, str]) -> Dict[str, str]:
    """Incremental FIRST-SEEN resolution. READ-ONLY by contract: reading a channel must not
    create a user row or bump anyone's `last_seen`."""
    self_name = sanitize_name(getattr(client, "bot_handle", None)
                              or (config.bot_name_aliases or ["assistant"])[0])
    pending: List[str] = []
    for message in messages:
        sender = message.sender_id
        if sender and sender not in actors:
            if message.sender_type == "self":
                actors[sender] = self_name
            elif message.sender_type == "other_bot":
                if message.raw_bot_name:
                    actors[sender] = message.raw_bot_name
            elif sender not in pending:
                pending.append(sender)
        for mention in message.mention_ids:
            if mention not in actors and mention not in pending:
                pending.append(mention)
    if pending:
        resolver = getattr(client, "resolve_usernames", None)
        api_client = getattr(getattr(client, "app", None), "client", None)
        if callable(resolver):
            try:
                resolved = await resolver(pending, api_client)
            except Exception as e:  # noqa: BLE001 — a name we cannot read is not a failed crawl
                logger.warning(f"crawl actor resolution failed: {e}")
                resolved = {}
            for uid, name in (resolved or {}).items():
                clean = sanitize_name(name)
                if clean:
                    actors[uid] = clean
    return actors


def _seed_roots(inventory: Dict[str, Dict[str, Any]], candidates: Iterable[str], *,
                high_ts: str) -> None:
    for raw in candidates or ():
        if not raw:
            continue
        root = str(raw)
        try:
            if parse_ts(root) > parse_ts(high_ts):
                continue
        except TimestampError:
            continue
        blank: Dict[str, Any] = {"root_ts": root, "done": False, "reply_cursor_ts": None,
                                 "reply_count": 0, "last_canonical_message_ts": None,
                                 "root_snippet_len": None, "observed_raw_pages": 0}
        inventory.setdefault(root, blank)


def seed_source_pin_roots(*, activity_roots: Sequence[str], receipt_roots: Sequence[str],
                          high_ts: str) -> Dict[str, Dict[str, Any]]:
    """Root discovery sources 3 and 4, SEEDED INSIDE THE SOURCE-PIN TRANSACTION.

    A pre-floor root with in-window replies is therefore present BY CONSTRUCTION: it arrives from
    the activity index and from receipts, not from a history walk that structurally cannot see it
    (the parent keeps its original ts, so `history(oldest=floor)` never surfaces it).
    """
    inventory: Dict[str, Dict[str, Any]] = {}
    _seed_roots(inventory, activity_roots, high_ts=high_ts)
    _seed_roots(inventory, receipt_roots, high_ts=high_ts)
    return inventory


async def run_phase_one(*, db: Any, client: Any, checkpoint: Dict[str, Any],
                        budget: SliceBudget, frozen_receipts: Mapping[str, str],
                        artifact_renders: Optional[Mapping[str, Sequence[
                            Mapping[str, Any]]]] = None,
                        marker_lines: Optional[Mapping[str, Sequence[str]]] = None,
                        page_limit: int = 200,
                        shutdown: Optional[asyncio.Event] = None) -> Dict[str, Any]:
    """INVENTORY: walk BOTH sources and persist the CANONICAL EVENT SKELETON.

    History walks BACKWARD from `pinned_H`, resumable by `inventory_cursor_ts`; replies walk
    FORWARD per root from that root's own exclusive `reply_cursor_ts`. History alone cannot
    inventory threaded events — `reply_count` supplies no timestamps and counts chrome, deleted,
    post-H and broadcast-duplicated events indiscriminately.

    Every page commits PAGE-ATOMICALLY: rows, roots, actors, density AND the cursor advance in one
    transaction, so neither a permanent gap nor replay ambiguity is reachable.
    """
    team_id = str(checkpoint["team_id"])
    channel_id = str(checkpoint["channel_id"])
    namespace = str(checkpoint["namespace"])
    crawl_id = str(checkpoint["crawl_id"])
    high = str(checkpoint["pinned_H"])
    floor_ts = str(checkpoint["input_floor_ts"])
    floor_inclusive = bool(int(checkpoint.get("input_floor_inclusive") or 0))
    renders = dict(artifact_renders or {})
    markers = dict(marker_lines or {})

    inventory: Dict[str, Dict[str, Any]] = {
        str(k): dict(v) for k, v in load_json_field(
            checkpoint.get("root_inventory"), {}).items()}
    density: List[Dict[str, Any]] = list(load_json_field(
        checkpoint.get("history_span_density"), []))
    actors: Dict[str, str] = dict(load_json_field(checkpoint.get("actor_snapshot"), {}))
    cursor = checkpoint.get("inventory_cursor_ts")

    async def _commit(rows: Sequence[SkeletonRow], patch: Dict[str, Any]) -> None:
        await db.commit_crawl_page_async(
            team_id=team_id, channel_id=channel_id, namespace=namespace, crawl_id=crawl_id,
            skeleton_rows=[r.as_row() for r in rows], checkpoint_patch=patch)
        # The in-memory tuple must match what committed: phase II is handed this same dict, and
        # a stale copy would size a chunk against an inventory the database has moved past.
        checkpoint.update(patch)

    def _row_for(message: NormalizedMessage, source_rank: int) -> SkeletonRow:
        return build_skeleton_row(message, actor_names=actors,
                                  artifact_renders=renders.get(str(message.ts), ()),
                                  marker_lines=markers.get(str(message.ts), ()),
                                  source_rank=source_rank)

    def _note_root(message: NormalizedMessage) -> None:
        root = message.root_ts
        _seed_roots(inventory, [root], high_ts=high)
        entry = inventory.get(str(root))
        if entry is not None and entry.get("root_snippet_len") is None and (
                str(message.ts) == str(root)):
            entry["root_snippet_len"] = _root_snippet_bytes(message)

    # --- history, BACKWARD -------------------------------------------------------------------
    # `inventory_cursor_ts` is NULL both before the walk starts and once it completes, so the
    # density list — appended per page — is what tells a resume which of the two it is.
    history_done = cursor is None and bool(density)
    while not history_done:
        if shutdown is not None and shutdown.is_set():
            return {"outcome": "deferred", "reason": "shutdown", "phase": PHASE_INVENTORY}
        if budget.exhausted():
            return {"outcome": "deferred", "reason": "budget", "phase": PHASE_INVENTORY}
        latest = str(cursor) if cursor else high
        page = await _history_page(client, channel_id=channel_id, latest=latest,
                                   inclusive=cursor is None, limit=page_limit)
        budget.charge_page()
        messages = _page_messages(client, page.messages, channel_id=channel_id,
                                  team_id=team_id, origin=ORIGIN_HISTORY)
        actors = await _resolve_new_actors(client, messages, actors)
        rows: List[SkeletonRow] = []
        page_oldest: Optional[str] = None
        reached_floor = not page.claims_more or not messages
        for message in messages:
            if page_oldest is None or parse_ts(message.ts) < parse_ts(page_oldest):
                page_oldest = str(message.ts)
            if not in_window(message.ts, floor_ts, floor_inclusive, high):
                if parse_ts(message.ts) <= parse_ts(floor_ts):
                    reached_floor = True
                continue
            if message.reply_count or message.latest_reply or message.is_reply:
                _note_root(message)
            if not is_skeleton_member(message, frozen_receipts=frozen_receipts):
                continue
            rows.append(_row_for(message, SOURCE_RANK_HISTORY))
        span_start = page_oldest or latest
        density.append({"span_start_ts": span_start, "span_end_ts": latest,
                        "observed_raw_pages": 1})
        cursor = None if reached_floor else page_oldest
        patch: Dict[str, Any] = {
            "inventory_cursor_ts": cursor,
            "root_inventory": dump_json_field(inventory),
            "history_span_density": dump_json_field(density),
            "actor_snapshot": dump_json_field(actors),
        }
        await _commit(rows, patch)
        checkpoint["inventory_cursor_ts"] = cursor
        history_done = reached_floor or not page.claims_more

    # --- replies, FORWARD per root ------------------------------------------------------------
    for root_ts in sorted(inventory, key=parse_ts):
        entry = inventory[root_ts]
        while not entry.get("done"):
            if shutdown is not None and shutdown.is_set():
                return {"outcome": "deferred", "reason": "shutdown", "phase": PHASE_INVENTORY}
            if budget.exhausted():
                return {"outcome": "deferred", "reason": "budget", "phase": PHASE_INVENTORY}
            reply_cursor = entry.get("reply_cursor_ts")
            page = await _replies_page(client, channel_id=channel_id, root_ts=root_ts,
                                       oldest=reply_cursor, high=high, limit=page_limit)
            budget.charge_page()
            messages = _page_messages(client, page.messages, channel_id=channel_id,
                                      team_id=team_id, origin=ORIGIN_REPLIES)
            actors = await _resolve_new_actors(client, messages, actors)
            rows = []
            newest = reply_cursor
            for message in messages:
                if str(message.ts) == str(root_ts) and entry.get("root_snippet_len") is None:
                    entry["root_snippet_len"] = _root_snippet_bytes(message)
                if reply_cursor and parse_ts(message.ts) <= parse_ts(reply_cursor):
                    continue
                if newest is None or parse_ts(message.ts) > parse_ts(newest):
                    newest = str(message.ts)
                if not in_window(message.ts, floor_ts, floor_inclusive, high):
                    continue
                if not is_skeleton_member(message, frozen_receipts=frozen_receipts):
                    continue
                rows.append(_row_for(message, SOURCE_RANK_REPLIES))
            entry["reply_cursor_ts"] = newest
            entry["observed_raw_pages"] = int(entry.get("observed_raw_pages") or 0) + 1
            entry["done"] = not page.claims_more or not messages or newest == reply_cursor
            await _commit(rows, {"root_inventory": dump_json_field(inventory),
                                 "actor_snapshot": dump_json_field(actors)})
        if entry.get("root_snippet_len") is None:
            entry["root_snippet_len"] = await _fetch_root_snippet(
                client, channel_id=channel_id, team_id=team_id, root_ts=root_ts)
            await _commit((), {"root_inventory": dump_json_field(inventory)})

    # --- freeze the actor snapshot and SEAL ---------------------------------------------------
    digest = actor_snapshot_hash(actors)
    sealed = await db.seal_event_skeleton_async(crawl_id)
    for root_ts, aggregate in (sealed.get("roots") or {}).items():
        blank: Dict[str, Any] = {"root_ts": str(root_ts), "done": True,
                                 "reply_cursor_ts": None, "reply_count": 0,
                                 "last_canonical_message_ts": None, "root_snippet_len": 0,
                                 "observed_raw_pages": 0}
        entry = inventory.setdefault(str(root_ts), blank)
        entry.update({k: v for k, v in aggregate.items() if k != "observed_raw_pages"})
    final_patch: Dict[str, Any] = {
        "phase": PHASE_CHUNKS,
        "chunk_index": 0,
        "inventory_cursor_ts": None,
        "actor_snapshot": dump_json_field(actors),
        "actor_snapshot_hash": digest,
        "root_inventory": dump_json_field(inventory),
        "history_span_density": dump_json_field(density),
        "event_count": int(sealed.get("events") or 0),
    }
    checkpoint.update(final_patch)
    await db.upsert_crawl_checkpoint_async(dict(checkpoint))
    return {"outcome": "sealed", "events": int(sealed.get("events") or 0),
            "phase": PHASE_CHUNKS}


async def _fetch_root_snippet(client: Any, *, channel_id: str, team_id: str,
                              root_ts: str) -> int:
    """ONE targeted fetch for a pre-floor root the walks never surfaced."""
    method = _web(client, "conversations_history")
    if method is None:
        return 0
    try:
        page = await fetch_page(method, {"channel": channel_id, "latest": root_ts,
                                         "oldest": root_ts, "inclusive": True, "limit": 1},
                                label="crawl root snippet")
    except HistoryFetchError as e:
        logger.warning(f"root snippet for {channel_id}/{root_ts} unavailable: {e}")
        return 0
    raw = next((m for m in page.messages if str(m.get("ts")) == str(root_ts)), None)
    if raw is None:
        return 0
    try:
        message = normalize_slack_message(client, raw, channel_id=channel_id,
                                          origin=ORIGIN_HISTORY, team_id=team_id,
                                          allow_any_subtype=True)
    except TimestampError:
        return 0
    return _root_snippet_bytes(message)


# ---------------------------------------------------------------- phase II

async def parent_chunk_zero(db: Any, checkpoint: Mapping[str, Any]) -> Optional[str]:
    """Chunk zero of an incremental crawl: the parent's PERSISTED payload bytes, VERBATIM.

    NAMED EXCEPTION — a restart may read the immutable parent generation rather than duplicating
    a potentially large payload into the checkpoint. The read is restart-safe precisely because
    generations are immutable. A missing parent, or one failing its `payload_hash` check, means
    DISCARD and restart as a raw crawl.
    """
    parent_id = checkpoint.get("parent_snapshot_id")
    getter = getattr(db, "get_snapshot_row_async", None)
    if not parent_id or not callable(getter):
        return None
    row = await getter(parent_id)
    if not row:
        raise SourceMutated(f"incremental parent {parent_id} is gone")
    payload = row.get("payload_bytes")
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if payload is None:
        raise SourceMutated(f"incremental parent {parent_id} carries no payload")
    stored = str(row.get("payload_hash") or "")
    if stored and hashlib.sha256(payload).hexdigest() != stored:
        raise SourceMutated(f"incremental parent {parent_id} failed its payload_hash check")
    return prior_summary_chunk(payload, old_boundary_ts=str(row.get("boundary_ts")))


async def run_phase_two(*, db: Any, client: Any, openai_client: Any,
                        checkpoint: Dict[str, Any], attempt: CompactionAttempt,
                        budget: SliceBudget, frozen_receipts: Mapping[str, str],
                        artifact_renders: Optional[Mapping[str, Sequence[
                            Mapping[str, Any]]]] = None,
                        map_bound: Optional[int] = None,
                        max_output_tokens: Optional[int] = None,
                        page_limit: int = 200,
                        shutdown: Optional[asyncio.Event] = None) -> Dict[str, Any]:
    """CHUNKS, oldest first. Chunk k is skeleton rows `[k*500, (k+1)*500)` exactly.

    THE CHUNK IS THE ATOMIC WORK UNIT: the budget is checked BETWEEN chunks, a started chunk
    always runs to completion even past the nominal budget, and a chunk whose known cost exceeds a
    whole worker budget still completes with a CRITICAL naming the cost — refusing it would mean
    the chunk never completes and the crawl never finishes.

    COORDINATOR SHUTDOWN is the one exception and uses CRASH SEMANTICS: nothing partial is
    persisted and the restart refetches that chunk from the beginning.
    """
    team_id = str(checkpoint["team_id"])
    channel_id = str(checkpoint["channel_id"])
    crawl_id = str(checkpoint["crawl_id"])
    high = str(checkpoint["pinned_H"])
    actors = load_json_field(checkpoint.get("actor_snapshot"), {})
    inventory = load_json_field(checkpoint.get("root_inventory"), {})
    density = load_json_field(checkpoint.get("history_span_density"), [])
    digests: List[str] = list(load_json_field(checkpoint.get("chunk_hashes"), []))
    aggregates: List[Dict[str, Any]] = list(load_json_field(
        checkpoint.get("chunk_aggregates"), []))
    summaries: Dict[str, str] = dict(load_json_field(checkpoint.get("chunk_summaries"), {}))
    renders = dict(artifact_renders or {})
    bound = map_bound if map_bound is not None else utility_map_bound()
    out_tokens = int(max_output_tokens if max_output_tokens is not None
                     else config.summary_byte_cap)
    total_events = int(checkpoint.get("event_count") or await db.skeleton_count_async(crawl_id))
    index = int(checkpoint.get("chunk_index") or 0)
    map_base = admission_charge(COMPACTION_MAP_PROMPT_V1) + 1

    # The PRIOR SUMMARY block is chunk index 0 in its entirety regardless of size, so event
    # chunks begin at index 1 for an incremental crawl.
    offset = 0
    if str(checkpoint.get("crawl_mode")) == CRAWL_MODE_INCREMENTAL:
        offset = 1
        if not digests:
            prior = await parent_chunk_zero(db, checkpoint)
            if prior is not None:
                digests.append(hashlib.sha256(chunk_bytes([prior])).hexdigest())
                aggregates.append({"seq_start": 0, "seq_end": 0, "events": 0,
                                   "cum_projection_charge": 0, "cum_base_charge": 0,
                                   "last_canonical_message_ts": None, "message_count": 0})
                summaries.update(await run_map_stage(
                    openai_client, attempt=attempt,
                    chunks=map_chunks(0, [prior], bound=bound, base_charge=map_base),
                    max_output_tokens=out_tokens, cache_key=f"compaction:{channel_id}"))

    while index * HASH_CHUNK_EVENTS < total_events:
        if shutdown is not None and shutdown.is_set():
            return {"outcome": "deferred", "reason": "shutdown", "chunk_index": index}
        if budget.exhausted():
            return {"outcome": "deferred", "reason": "budget", "chunk_index": index}
        rows = await db.skeleton_slice_async(crawl_id, index * HASH_CHUNK_EVENTS,
                                             (index + 1) * HASH_CHUNK_EVENTS)
        if not rows:
            break
        plan = chunk_refetch_plan(rows, root_inventory=inventory,
                                  history_span_density=density)
        if plan["estimated_pages"] > budget.page_budget:
            logger.critical(
                f"compaction crawl {channel_id} chunk {index} costs ~{plan['estimated_pages']} "
                f"pages, more than a whole {budget.page_budget}-page worker budget; running it "
                "to completion anyway so the crawl can finish")
        try:
            batches = await _refetch_chunk(client, channel_id=channel_id, team_id=team_id,
                                           plan=plan, high=high, page_limit=page_limit,
                                           budget=budget)
            events = [m for m in merge_events(batches)
                      if is_skeleton_member(m, frozen_receipts=frozen_receipts)
                      and in_window(m.ts, plan["span_start_ts"], True, plan["span_end_ts"])]
            projections = verify_slice(events, rows, actor_names=actors,
                                       renders_for=lambda ts: renders.get(ts, ()))
            chunks = map_chunks(index + offset, projections, bound=bound,
                                base_charge=map_base)
            produced = await run_map_stage(openai_client, attempt=attempt, chunks=chunks,
                                           max_output_tokens=out_tokens,
                                           cache_key=f"compaction:{channel_id}")
        except asyncio.CancelledError:
            # Crash semantics: no partial chunk is persisted, and the restart refetches it.
            logger.info(f"compaction crawl {channel_id} chunk {index} cancelled mid-flight")
            raise
        digests.append(hashlib.sha256(chunk_bytes(projections)).hexdigest())
        summaries.update(produced)
        previous: Mapping[str, Any] = aggregates[-1] if aggregates else {}
        aggregates.append({
            "seq_start": index * HASH_CHUNK_EVENTS,
            "seq_end": index * HASH_CHUNK_EVENTS + len(rows),
            "events": len(rows),
            "cum_projection_charge": int(previous.get("cum_projection_charge") or 0)
            + sum(int(_skeleton_field(r, "projected_byte_len")) for r in rows),
            "cum_base_charge": int(previous.get("cum_base_charge") or 0)
            + sum(int(_skeleton_field(r, "base_canonical_bytes")) for r in rows),
            "last_canonical_message_ts": str(rows[-1]["ts"]),
            "message_count": len(rows),
        })
        index += 1
        checkpoint.update({
            "chunk_index": index,
            "chunk_hashes": dump_json_field(digests),
            "chunk_aggregates": dump_json_field(aggregates),
            "chunk_summaries": dump_json_field(summaries),
            **attempt.checkpoint_patch(),
        })
        await db.upsert_crawl_checkpoint_async(dict(checkpoint))
    return {"outcome": "chunks_complete", "chunk_index": index, "digests": digests,
            "aggregates": aggregates, "summaries": summaries}


async def _refetch_chunk(client: Any, *, channel_id: str, team_id: str, plan: Mapping[str, Any],
                         high: str, page_limit: int,
                         budget: SliceBudget) -> List[List[NormalizedMessage]]:
    batches: List[List[NormalizedMessage]] = []
    span_end = str(plan["span_end_ts"])
    span_start = str(plan["span_start_ts"])
    cursor: Optional[str] = None
    while True:
        page = await _history_page(client, channel_id=channel_id,
                                   latest=str(cursor) if cursor else span_end,
                                   inclusive=cursor is None, limit=page_limit)
        budget.charge_page()
        messages = _page_messages(client, page.messages, channel_id=channel_id,
                                  team_id=team_id, origin=ORIGIN_HISTORY)
        batches.append(messages)
        oldest = min((m.ts for m in messages), key=parse_ts, default=None)
        if (not page.claims_more or oldest is None or cursor == oldest
                or parse_ts(oldest) <= parse_ts(span_start)):
            break
        cursor = oldest
    for root_ts in plan["roots"]:
        cursor = None
        while True:
            page = await _replies_page(client, channel_id=channel_id, root_ts=root_ts,
                                       oldest=cursor or span_start, high=high, limit=page_limit)
            budget.charge_page()
            messages = _page_messages(client, page.messages, channel_id=channel_id,
                                      team_id=team_id, origin=ORIGIN_REPLIES)
            batches.append(messages)
            newest = max((m.ts for m in messages), key=parse_ts, default=None)
            if not page.claims_more or newest is None or newest == cursor:
                break
            cursor = newest
    return batches


# ---------------------------------------------------------------- boundary-chunk regeneration

async def regenerate_boundary_chunk(*, db: Any, client: Any, openai_client: Any,
                                    checkpoint: Mapping[str, Any],
                                    attempt: CompactionAttempt, boundary_ts: str,
                                    frozen_receipts: Mapping[str, str],
                                    artifact_renders: Optional[Mapping[str, Sequence[
                                        Mapping[str, Any]]]] = None,
                                    map_bound: Optional[int] = None,
                                    max_output_tokens: Optional[int] = None,
                                    page_limit: int = 200,
                                    budget: Optional[SliceBudget] = None) -> Dict[str, Any]:
    """The boundary chunk is regenerated by the ORDINARY chunk operation over its truncated span.

    Chunks wholly below the boundary keep their digests and summaries byte-identical; chunks
    wholly above it are dropped; the final partial chunk contributes a digest over ONLY its
    INCLUDED PREFIX — `source_hash` covers `[source_floor_ts, boundary_ts]`, not `pinned_H`.
    """
    crawl_id = str(checkpoint["crawl_id"])
    team_id = str(checkpoint["team_id"])
    channel_id = str(checkpoint["channel_id"])
    actors = load_json_field(checkpoint.get("actor_snapshot"), {})
    inventory = load_json_field(checkpoint.get("root_inventory"), {})
    density = load_json_field(checkpoint.get("history_span_density"), [])
    digests: List[str] = list(load_json_field(checkpoint.get("chunk_hashes"), []))
    aggregates: List[Dict[str, Any]] = list(load_json_field(
        checkpoint.get("chunk_aggregates"), []))
    summaries: Dict[str, str] = dict(load_json_field(checkpoint.get("chunk_summaries"), {}))
    renders = dict(artifact_renders or {})
    slice_budget = budget or SliceBudget()
    bound = map_bound if map_bound is not None else utility_map_bound()
    out_tokens = int(max_output_tokens if max_output_tokens is not None
                     else config.summary_byte_cap)

    boundary_chunk = None
    for entry in aggregates:
        if int(entry.get("events") or 0) <= 0:
            continue
        last = entry.get("last_canonical_message_ts")
        if last and parse_ts(last) >= parse_ts(boundary_ts):
            boundary_chunk = entry
            break
    if boundary_chunk is None:
        return {"digests": digests, "summaries": summaries, "regenerated": None}

    offset = 1 if str(checkpoint.get("crawl_mode")) == CRAWL_MODE_INCREMENTAL else 0
    index = int(boundary_chunk["seq_start"]) // HASH_CHUNK_EVENTS + offset
    rows = await db.skeleton_slice_async(crawl_id, int(boundary_chunk["seq_start"]),
                                         int(boundary_chunk["seq_end"]))
    included = [r for r in rows if parse_ts(str(r["ts"])) <= parse_ts(boundary_ts)]
    plan = chunk_refetch_plan(included, root_inventory=inventory,
                              history_span_density=density)
    batches = await _refetch_chunk(client, channel_id=channel_id, team_id=team_id, plan=plan,
                                  high=boundary_ts, page_limit=page_limit,
                                  budget=slice_budget)
    events = [m for m in merge_events(batches)
              if is_skeleton_member(m, frozen_receipts=frozen_receipts)
              and in_window(m.ts, plan["span_start_ts"], True, plan["span_end_ts"])]
    projections = verify_slice(events, included, actor_names=actors,
                               renders_for=lambda ts: renders.get(ts, ()))
    map_base = admission_charge(COMPACTION_MAP_PROMPT_V1) + 1
    produced = await run_map_stage(openai_client, attempt=attempt,
                                   chunks=map_chunks(index, projections, bound=bound,
                                                     base_charge=map_base),
                                   max_output_tokens=out_tokens,
                                   cache_key=f"compaction:{channel_id}")
    kept = {key: value for key, value in summaries.items()
            if summary_key(key)[0] < index}
    kept.update(produced)
    digests = digests[:index] + [hashlib.sha256(chunk_bytes(projections)).hexdigest()]
    return {"digests": digests, "summaries": kept, "regenerated": index,
            "included_events": len(included)}


# ---------------------------------------------------------------- lineage

def inherited_manifest(parent_rows: Sequence[Mapping[str, Any]],
                       own_rows: Sequence[Mapping[str, Any]], *,
                       snapshot_id: str) -> List[Dict[str, Any]]:
    """A descendant's manifest = the parent's rows CARRIED FORWARD (retaining the PARENT's
    `content_hash` and `status_at_capture`, so a later same-row change is still detected) UNION
    the rows derived from its own projection span, its own row winning on a collision.

    Without inheritance every artifact already inside the parent summary would reappear as "late"
    evidence forever.
    """
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in parent_rows or ():
        key = (str(row.get("artifact_namespace")), str(row.get("row_id")))
        merged[key] = {**dict(row), "snapshot_id": snapshot_id}
    for row in own_rows or ():
        key = (str(row.get("artifact_namespace")), str(row.get("row_id")))
        merged[key] = {**dict(row), "snapshot_id": snapshot_id}
    return [merged[key] for key in sorted(merged)]


def manifest_rows_from_projection(captured: Iterable[Mapping[str, Any]], *,
                                  snapshot_id: str) -> List[Dict[str, Any]]:
    """The manifest is exactly the set whose RENDER BYTES the projection included. An entry the
    projection omitted for any reason stays post-breakpoint late-artifact evidence."""
    rows: List[Dict[str, Any]] = []
    for entry in captured or ():
        rows.append({
            "snapshot_id": snapshot_id,
            "artifact_namespace": str(entry.get("artifact_namespace")),
            "row_id": str(entry.get("row_id")),
            "source_ts": str(entry.get("source_ts")),
            "captured_render_version": str(entry.get("captured_render_version") or "1"),
            "content_hash": str(entry.get("content_hash")),
            "status_at_capture": str(entry.get("status_at_capture") or "complete"),
        })
    return sorted(rows, key=lambda r: (r["artifact_namespace"], r["row_id"]))


async def publish_stale_retained(*, db: Any, coordinator: Any, parent: Mapping[str, Any],
                                 team_id: str, channel_id: str, namespace: str,
                                 serializer_version: int, sizing: Mapping[str, Any],
                                 headroom_source: str, headroom_tokens: int,
                                 fit: str, expected_previous_id: Optional[str],
                                 crawl_id: Optional[str] = None,
                                 attempt_seq: int = 0,
                                 now: Optional[float] = None,
                                 satisfy: Optional[Dict[str, Any]] = None,
                                 dormancy: Optional[Dict[str, Any]] = None
                                 ) -> Dict[str, Any]:
    """The §1f STALE-RETAINED REPLACEMENT — a COPY, not a regeneration.

    Payload bytes are the parent's plus the stale marker at its pinned A2 position; anchors,
    manifest, `source_hash`, `source_floor_ts` and `boundary_ts` are INHERITED; the manifest AND
    anchor-provenance rows are COPIED; FRESH FRONTIERS are pinned at copy time — inheriting the
    parent's would make the replacement unpublishable, because the very observations that
    invalidated the parent sit above them. SIZING EVIDENCE IS RECOMPUTED, NEVER INHERITED.

    No model call happens, but it STILL emits `op=build` with `status="copied"` and
    `call_count=0`: the ledger checker requires a build before every publish.
    """
    if str(parent.get("status_result") or "") == "payload_corrupt":
        return {"outcome": "failed", "reason": "payload_corrupt_parent", "snapshot_id": None,
                "fit_result": None}
    payload = parent.get("payload_bytes")
    if payload is None:
        return {"outcome": "failed", "reason": "payload_missing", "snapshot_id": None,
                "fit_result": None}
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    boundary_ts = str(parent.get("boundary_ts"))
    marked = stale_marked_payload(payload, boundary_ts=boundary_ts)

    frontier = await _max_observation_id(db, team_id, channel_id)
    snapshot_id = uuid.uuid4().hex
    manifest = [{**dict(row), "snapshot_id": snapshot_id}
                for row in await _rows(db, "snapshot_manifest_async",
                                       parent.get("snapshot_id"))]
    anchors = [{**dict(row), "snapshot_id": snapshot_id,
                "observation_frontier": int(frontier)}
               for row in await _rows(db, "snapshot_anchor_provenance_async",
                                      parent.get("snapshot_id"))]

    attempt = CompactionAttempt(
        crawl_id=str(crawl_id or uuid.uuid4().hex), attempt_seq=int(attempt_seq),
        team_id=team_id, channel_id=channel_id, namespace=namespace,
        model=str(sizing.get("model") or config.utility_model))
    snapshot = {
        "snapshot_id": snapshot_id, "team_id": team_id, "channel_id": channel_id,
        "namespace": namespace, "serializer_version": int(serializer_version),
        "status": "candidate", "boundary_ts": boundary_ts,
        "source_floor_ts": str(parent.get("source_floor_ts")),
        "parent_snapshot_id": parent.get("snapshot_id"),
        "prompt_version": PROMPT_VERSION,
        "model": attempt.model,
        "source_hash": parent.get("source_hash"),
        "payload_bytes": marked,
        "payload_hash": hashlib.sha256(marked).hexdigest(),
        "anchor_payload_bytes": parent.get("anchor_payload_bytes"),
        "mutation_frontier": int(frontier),
        "headroom_source": headroom_source, "headroom_tokens": int(headroom_tokens),
        "effective_window": int(sizing.get("window") or 0),
        "sizing_profile": str(sizing.get("sizing_profile") or ""),
        "fit_result": fit,
    }
    candidate_id = await db.insert_compaction_candidate_async(
        snapshot=snapshot, manifest_rows=manifest, anchor_rows=anchors)
    at = time.time() if now is None else float(now)
    build = attempt.build_body(status=BUILD_COPIED, at=at)
    result = await coordinator.publish(
        team_id=team_id, channel_id=channel_id, namespace=namespace,
        serializer_version=int(serializer_version), snapshot_id=candidate_id,
        expected_previous_id=expected_previous_id,
        source_floor_ts=str(snapshot["source_floor_ts"]), boundary_ts=boundary_ts,
        mutation_frontier=int(frontier),
        status="published_stale",
        outbox_rows=[attempt.outbox_row(build),
                     attempt.outbox_row(attempt.publish_body(
                         at=at, snapshot_id=candidate_id, generation=0,
                         boundary_ts=boundary_ts, fit=fit,
                         serializer_version=int(serializer_version)))],
        satisfy=satisfy, dormancy=dormancy)
    if not result.get("won"):
        return {"outcome": "discarded", "reason": result.get("reason"),
                "snapshot_id": candidate_id, "fit_result": fit}
    return {"outcome": "published", "snapshot_id": candidate_id, "fit_result": fit,
            "reason": None, "generation": result.get("generation")}


async def _rows(db: Any, method: str, snapshot_id: Any) -> List[Mapping[str, Any]]:
    getter = getattr(db, method, None)
    if not callable(getter) or not snapshot_id:
        return []
    return list(await getter(snapshot_id) or [])


# ---------------------------------------------------------------- completion

@dataclass(frozen=True)
class Measurement:
    """What the §1e postcondition is remeasured against, after generation."""
    summary_block: str
    anchor_block: str
    total_charge: int
    fit: str


def measure_request(*, summary_block: str, horizon_bytes: int, retained_charge: int,
                    retained_items: int, non_compactable_tokens: int,
                    target_tokens: int, trigger_tokens: int,
                    item_overhead: int = ITEM_STRUCTURAL_OVERHEAD) -> Measurement:
    """The COMPLETE request's admission charge, with the produced summary in it.

    Every term is in the admitted currency — one token per UTF-8 byte — and the summary and
    horizon items each carry their own structural overhead.
    """
    total = (admission_charge(summary_block) + admission_charge_bytes(horizon_bytes)
             + int(retained_charge) + (int(retained_items) + 2) * int(item_overhead)
             + int(non_compactable_tokens))
    return Measurement(summary_block=summary_block, anchor_block="", total_charge=total,
                       fit=fit_result(total_charge=total, target_tokens=target_tokens,
                                      trigger_tokens=trigger_tokens))


async def complete_attempt(*, db: Any, client: Any, openai_client: Any, coordinator: Any,
                           checkpoint: Mapping[str, Any], attempt: CompactionAttempt,
                           sizing: Mapping[str, Any], headroom_source: str,
                           headroom_tokens: int, expected_previous_id: Optional[str],
                           serializer_version: int,
                           horizon_bytes: int = 0,
                           captured_artifacts: Sequence[Mapping[str, Any]] = (),
                           parent_manifest: Sequence[Mapping[str, Any]] = (),
                           parent_payload_bytes: Optional[bytes] = None,
                           map_bound: Optional[int] = None,
                           max_output_tokens: Optional[int] = None,
                           max_generation_attempts: int = 2,
                           budget: Optional[SliceBudget] = None,
                           satisfy: Optional[Dict[str, Any]] = None,
                           dormancy: Optional[Dict[str, Any]] = None,
                           now: Optional[float] = None) -> Dict[str, Any]:
    """Boundary → anchors → generation → validation → insertion → publication.

    NOTHING IS INSERTED UNTIL VALIDATION PASSES: the candidate row, its manifest rows and its
    anchor-provenance rows go in ONE transaction AFTER output validation succeeds, so an ordinary
    discarded candidate has no rows to clean up.
    """
    team_id = str(checkpoint["team_id"])
    channel_id = str(checkpoint["channel_id"])
    namespace = str(checkpoint["namespace"])
    crawl_id = str(checkpoint["crawl_id"])
    receipts = load_json_field(checkpoint.get("frozen_receipts"), {})
    renders = load_json_field(checkpoint.get("frozen_renders"), {})
    inventory = load_json_field(checkpoint.get("root_inventory"), {})
    aggregates = load_json_field(checkpoint.get("chunk_aggregates"), [])
    digests: List[str] = list(load_json_field(checkpoint.get("chunk_hashes"), []))
    summaries: Dict[str, str] = dict(load_json_field(checkpoint.get("chunk_summaries"), {}))
    events = int(checkpoint.get("event_count") or 0)
    rows = list(await db.skeleton_slice_async(crawl_id, 0, events) or [])
    snippets = {str(k): int(v.get("root_snippet_len") or 0) for k, v in inventory.items()}

    # The RECEIPT-CONSTRAINED CAP comes FIRST: an old in_flight receipt can force the legal
    # boundary far below anything size would suggest, and finding it afterwards would mean
    # discarding the candidate.
    cap_ts = receipt_constrained_cap(receipts)
    target_tokens = int(sizing.get("target_tokens") or 0)
    trigger_tokens = int(sizing.get("trigger_tokens") or 0)
    reserve = int(config.summary_byte_cap) + int(horizon_bytes)
    tail_budget = target_tokens - int(headroom_tokens) - reserve

    from_index = 0
    outcome: Dict[str, Any] = {"outcome": "publish_nothing", "reason": "no_fit",
                               "snapshot_id": None, "fit_result": None}
    best: Optional[Tuple[Measurement, BoundaryCandidate, str, List[AnchorEntry], str,
                         str]] = None

    for round_index in range(2):  # one bounded more-aggressive retry (§1e)
        candidate = select_boundary(rows=rows, root_snippet_lens=snippets,
                                    budget_tokens=tail_budget,
                                    cap_ts=cap_ts, high_ts=str(checkpoint["pinned_H"]),
                                    prior_boundary_ts=checkpoint.get("input_floor_ts")
                                    if str(checkpoint.get("crawl_mode")) ==
                                    CRAWL_MODE_INCREMENTAL else None,
                                    chunk_aggregates=aggregates, from_index=from_index)
        if candidate is None:
            break

        regenerated = await regenerate_boundary_chunk(
            db=db, client=client, openai_client=openai_client, checkpoint=checkpoint,
            attempt=attempt, boundary_ts=candidate.boundary_ts, frozen_receipts=receipts,
            artifact_renders=renders, map_bound=map_bound,
            max_output_tokens=max_output_tokens, budget=budget)
        level_digests = regenerated["digests"] or digests
        level_summaries = regenerated["summaries"] or summaries
        source_hash = finish_source_hash(level_digests)

        roots, omitted = straddling_roots(inventory, boundary_ts=candidate.boundary_ts)
        anchors = await resolve_anchors(db=db, client=client, team_id=team_id,
                                        channel_id=channel_id, roots=roots,
                                        frozen_receipts=receipts)
        anchor_block = render_anchor_block([a.render_entry() for a in anchors],
                                           omitted=omitted)
        room = summary_room_tokens(
            target_tokens=target_tokens, headroom_tokens=int(headroom_tokens),
            shell_bytes=summary_shell_bytes(boundary_ts=candidate.boundary_ts),
            anchor_bytes=anchor_block_bytes(boundary_ts=candidate.boundary_ts,
                                            anchor_block=anchor_block),
            horizon_bytes=int(horizon_bytes),
            item_count=candidate.retained_items + 2,
            retained_charge=candidate.retained_charge)
        cap = summary_byte_cap(room_tokens=room)

        payload = ""
        failure: Optional[str] = None
        for _try in range(max(1, int(max_generation_attempts))):
            produced = await hierarchical_reduce(
                openai_client, attempt=attempt,
                summaries=ordered_summaries(level_summaries),
                bound=map_bound if map_bound is not None else utility_map_bound(),
                budget_tokens=max(1, cap // 4),
                max_output_tokens=int(max_output_tokens if max_output_tokens is not None
                                      else cap or config.summary_byte_cap))
            # Escape FIRST: A7 can LENGTHEN a payload (the "· " prefix on a forged marker), so
            # validating the raw output would cap bytes that are not the ones we persist.
            payload = _escape_payload(produced)
            failure = validate_summary(payload, cap=cap)
            if failure is None:
                break
        if failure is not None:
            outcome = {"outcome": "discarded", "reason": failure, "snapshot_id": None,
                       "fit_result": None}
            break

        block = render_summary_block(boundary_ts=candidate.boundary_ts, payload=payload,
                                     anchor_block=anchor_block, stale=False)
        measured = measure_request(
            summary_block=block, horizon_bytes=int(horizon_bytes),
            retained_charge=candidate.retained_charge,
            retained_items=candidate.retained_items,
            non_compactable_tokens=int(headroom_tokens),
            target_tokens=target_tokens, trigger_tokens=trigger_tokens)
        if best is None or measured.total_charge < best[0].total_charge:
            best = (measured, candidate, source_hash, anchors, block, anchor_block)
        if measured.fit == FIT_UNDER_TARGET:
            break
        if round_index == 0:
            # Bounded, more aggressive: push the boundary past the one that overshot.
            from_index = candidate.index + 1
            tail_budget = max(0, tail_budget - (measured.total_charge - target_tokens))
            continue

    if best is None:
        return outcome
    measured, candidate, source_hash, anchors, block, anchor_block = best
    if measured.fit == FIT_NONE:
        # Publish NOTHING and let the turn fail closed over-budget, honestly.
        return {"outcome": "publish_nothing", "reason": "over_trigger", "snapshot_id": None,
                "fit_result": None}

    payload_bytes = block.encode("utf-8")
    snapshot_id = uuid.uuid4().hex
    manifest = inherited_manifest(
        parent_manifest,
        manifest_rows_from_projection(captured_artifacts, snapshot_id=snapshot_id),
        snapshot_id=snapshot_id)
    anchor_rows = [a.provenance_row(team_id=team_id) for a in anchors]
    for row in anchor_rows:
        row["snapshot_id"] = snapshot_id
    snapshot = {
        "snapshot_id": snapshot_id, "team_id": team_id, "channel_id": channel_id,
        "namespace": namespace, "serializer_version": int(serializer_version),
        "status": "candidate", "boundary_ts": candidate.boundary_ts,
        "source_floor_ts": str(checkpoint["source_floor_ts"]),
        "parent_snapshot_id": checkpoint.get("parent_snapshot_id"),
        "prompt_version": PROMPT_VERSION, "model": attempt.model,
        "source_hash": source_hash,
        "payload_bytes": payload_bytes,
        "payload_hash": hashlib.sha256(payload_bytes).hexdigest(),
        "anchor_payload_bytes": anchor_block.encode("utf-8"),
        "mutation_frontier": int(checkpoint.get("mutation_frontier") or 0),
        "headroom_source": headroom_source, "headroom_tokens": int(headroom_tokens),
        "effective_window": int(sizing.get("window") or 0),
        "sizing_profile": str(sizing.get("sizing_profile") or ""),
        "fit_result": measured.fit,
    }
    candidate_id = await db.insert_compaction_candidate_async(
        snapshot=snapshot, manifest_rows=manifest, anchor_rows=anchor_rows)

    at = time.time() if now is None else float(now)
    build = attempt.build_body(status=BUILD_OK, at=at)
    result = await coordinator.publish(
        team_id=team_id, channel_id=channel_id, namespace=namespace,
        serializer_version=int(serializer_version), snapshot_id=candidate_id,
        expected_previous_id=expected_previous_id,
        source_floor_ts=str(checkpoint["source_floor_ts"]),
        boundary_ts=candidate.boundary_ts,
        mutation_frontier=int(checkpoint.get("mutation_frontier") or 0),
        outbox_rows=[attempt.outbox_row(build),
                     attempt.outbox_row(attempt.publish_body(
                         at=at, snapshot_id=candidate_id, generation=0,
                         boundary_ts=candidate.boundary_ts, fit=measured.fit,
                         serializer_version=int(serializer_version)))],
        satisfy=satisfy, dormancy=dormancy)
    if not result.get("won"):
        return {"outcome": "discarded", "reason": result.get("reason"),
                "snapshot_id": candidate_id, "fit_result": measured.fit}
    return {"outcome": "published", "snapshot_id": candidate_id,
            "fit_result": measured.fit, "reason": None,
            "generation": result.get("generation"),
            "boundary_ts": candidate.boundary_ts, "source_hash": source_hash}


# ---------------------------------------------------------------- entry points

def new_checkpoint(*, team_id: str, channel_id: str, namespace: str, pinned_H: str,
                   mutation_frontier: int, source_floor_ts: str, input_floor_ts: str,
                   input_floor_inclusive: bool, crawl_mode: str, serializer_version: int,
                   serializer_config_hash: str, sizing: Mapping[str, Any],
                   headroom_source: str, headroom_tokens: int,
                   parent_snapshot_id: Optional[str] = None,
                   frozen_renders: Optional[Mapping[str, Any]] = None,
                   frozen_receipts: Optional[Mapping[str, str]] = None,
                   root_inventory: Optional[Mapping[str, Any]] = None,
                   crawl_id: Optional[str] = None) -> Dict[str, Any]:
    """A fresh checkpoint tuple (§1n). `crawl_id` is a GLOBALLY UNIQUE uuid4 hex: that one
    property is what lets the skeleton, the outbox and the ledger identity key on it alone."""
    return {
        "team_id": team_id, "channel_id": channel_id, "namespace": namespace,
        "crawl_id": crawl_id or uuid.uuid4().hex,
        "crawl_mode": crawl_mode, "phase": PHASE_INVENTORY,
        "pinned_H": str(pinned_H), "mutation_frontier": int(mutation_frontier),
        "source_floor_ts": str(source_floor_ts), "input_floor_ts": str(input_floor_ts),
        "input_floor_inclusive": 1 if input_floor_inclusive else 0,
        "parent_snapshot_id": parent_snapshot_id, "boundary_ts": None,
        "serializer_version": int(serializer_version),
        "serializer_config_hash": str(serializer_config_hash),
        "prompt_version": PROMPT_VERSION,
        "sizing_profile": str(sizing.get("sizing_profile") or ""),
        "headroom_source": headroom_source, "headroom_tokens": int(headroom_tokens),
        "profile_version": profile_version(headroom_source=headroom_source,
                                           headroom_tokens=headroom_tokens),
        "inventory_cursor_ts": None,
        "root_inventory": dump_json_field(dict(root_inventory or {})),
        "history_span_density": dump_json_field([]),
        "actor_snapshot": dump_json_field({}), "actor_snapshot_hash": "",
        "chunk_index": 0, "chunk_hashes": dump_json_field([]),
        "chunk_aggregates": dump_json_field([]), "chunk_summaries": dump_json_field({}),
        "frozen_renders": dump_json_field(dict(frozen_renders or {})),
        "frozen_receipts": dump_json_field(dict(frozen_receipts or {})),
        "attempt_seq": 0, "attempt_tokens_in": 0, "attempt_tokens_out": 0,
        "attempt_cached_input_tokens": 0, "attempt_call_count": 0,
        "event_count": 0, "consecutive_discards": 0, "next_attempt_after": None,
        "updated_at": str(time.time()),
    }


async def resume_or_reset(*, db: Any, checkpoint: Dict[str, Any], live: Mapping[str, Any],
                          now: Optional[float] = None) -> Dict[str, Any]:
    """The §1n resume ladder, in order: backoff, config/version reset, mutation discard, resume."""
    if backoff_active(checkpoint, now=now):
        return {"action": "wait", "reason": "next_attempt_after"}
    reason = reset_reason(checkpoint, live)
    if reason is None and not verify_actor_snapshot(checkpoint):
        reason = "actor_snapshot_hash"
    if reason is not None:
        frontier = await _max_observation_id(db, str(checkpoint["team_id"]),
                                             str(checkpoint["channel_id"]))
        return {"action": "reset", "reason": reason,
                "checkpoint": apply_config_reset(checkpoint, live=live,
                                                 mutation_frontier=frontier)}
    observation = await mutation_discard_needed(db, checkpoint)
    if observation is not None:
        frontier = await _max_observation_id(db, str(checkpoint["team_id"]),
                                             str(checkpoint["channel_id"]))
        patch = apply_mutation_discard(checkpoint, mutation_frontier=frontier, now=now)
        if int(patch["consecutive_discards"]) >= STALL_DISCARD_THRESHOLD:
            logger.critical(
                f"compaction crawl {checkpoint['channel_id']} discarded "
                f"{patch['consecutive_discards']} times in a row; causing observation "
                f"{observation.get('id')} at {observation.get('subject_ts')}; backing off 1h")
        return {"action": "discard", "reason": "mutation", "observation": observation,
                "checkpoint": patch}
    return {"action": "resume", "reason": None, "checkpoint": checkpoint}


async def run_crawl_slice(*, db: Any, client: Any, team_id: str, channel_id: str,
                          namespace: str, coordinator: Any, trigger: str,
                          headroom_source: str, headroom_tokens: int,
                          budget: Optional[SliceBudget] = None,
                          openai_client: Any = None,
                          live: Optional[Mapping[str, Any]] = None,
                          shutdown: Optional[asyncio.Event] = None,
                          sizing: Optional[Mapping[str, Any]] = None,
                          expected_previous_id: Optional[str] = None,
                          serializer_version: int = 2,
                          completion: Optional[Mapping[str, Any]] = None,
                          now: Optional[float] = None) -> Dict[str, Any]:
    """ONE WORKER SLICE of background crawl work.

    Returns without publishing whenever the slice budget runs out — exhausting it reschedules and
    is not a failure, and it does not end the attempt. A slice that finishes the last chunk goes
    straight on to boundary selection, anchors, generation and publication.
    """
    checkpoint = await db.load_crawl_checkpoint_async(team_id, channel_id, namespace)
    if not checkpoint:
        return {"outcome": "failed", "reason": "no_checkpoint", "snapshot_id": None,
                "fit_result": None}
    checkpoint = dict(checkpoint)
    slice_budget = budget or SliceBudget()
    # Built BEFORE the resume ladder: a discard ENDS the attempt (§1l), and the `op=build` it
    # owes carries the calls already spent — which live on the checkpoint being discarded.
    attempt = CompactionAttempt.from_checkpoint(checkpoint, model=config.utility_model)
    at = time.time() if now is None else float(now)

    def _discarded(reason: str, **extra: Any) -> Dict[str, Any]:
        """The coordinator commits this row in the SAME transaction as the state change; the
        builder never writes telemetry itself."""
        body = attempt.build_body(status=BUILD_DISCARDED, at=at, reason=reason)
        return {"outcome": "discarded", "reason": reason, "snapshot_id": None,
                "fit_result": None, "outbox_rows": [attempt.outbox_row(body)], **extra}

    if live:
        decision = await resume_or_reset(db=db, checkpoint=checkpoint, live=live, now=now)
        if decision["action"] == "wait":
            return {"outcome": "deferred", "reason": decision["reason"], "snapshot_id": None,
                    "fit_result": None}
        if decision["action"] in ("reset", "discard"):
            checkpoint = decision["checkpoint"]
            await db.upsert_crawl_checkpoint_async(dict(checkpoint))
            return _discarded(str(decision["reason"]),
                              attempt_seq=int(checkpoint.get("attempt_seq") or 0))

    receipts = load_json_field(checkpoint.get("frozen_receipts"), {})
    renders = load_json_field(checkpoint.get("frozen_renders"), {})

    if int(checkpoint.get("phase") or PHASE_INVENTORY) == PHASE_INVENTORY:
        result = await run_phase_one(db=db, client=client, checkpoint=checkpoint,
                                     budget=slice_budget, frozen_receipts=receipts,
                                     artifact_renders=renders, shutdown=shutdown)
        if result["outcome"] == "deferred":
            return {"outcome": "deferred", "reason": result["reason"], "snapshot_id": None,
                    "fit_result": None}

    try:
        result = await run_phase_two(db=db, client=client, openai_client=openai_client,
                                     checkpoint=checkpoint, attempt=attempt,
                                     budget=slice_budget, frozen_receipts=receipts,
                                     artifact_renders=renders, shutdown=shutdown)
    except SourceMutated as e:
        frontier = await _max_observation_id(db, team_id, channel_id)
        patch = apply_mutation_discard(checkpoint, mutation_frontier=frontier, now=now)
        if int(patch["consecutive_discards"]) >= STALL_DISCARD_THRESHOLD:
            logger.critical(
                f"compaction crawl {channel_id} discarded {patch['consecutive_discards']} times "
                f"in a row; cause: {e}; backing off 1h")
        await db.upsert_crawl_checkpoint_async(dict(patch))
        return _discarded("mutation", subject_ts=e.subject_ts,
                          attempt_seq=int(patch.get("attempt_seq") or 0))
    if result["outcome"] == "deferred":
        return {"outcome": "deferred", "reason": result["reason"], "snapshot_id": None,
                "fit_result": None}

    resolved = sizing or resolve_sizing(
        model=config.utility_model,
        window=config.get_model_token_limit(config.utility_model))
    return await complete_attempt(
        db=db, client=client, openai_client=openai_client, coordinator=coordinator,
        checkpoint=checkpoint, attempt=attempt, sizing=resolved,
        headroom_source=headroom_source, headroom_tokens=headroom_tokens,
        expected_previous_id=expected_previous_id, serializer_version=serializer_version,
        budget=slice_budget, now=now, **dict(completion or {}))


async def run_incremental(*, db: Any, client: Any, team_id: str, channel_id: str,
                          namespace: str, coordinator: Any, parent: Mapping[str, Any],
                          h: str, headroom_source: str, headroom_tokens: int,
                          serializer_version: int = 2,
                          serializer_config_hash: str = "",
                          sizing: Optional[Mapping[str, Any]] = None,
                          frozen_renders: Optional[Mapping[str, Any]] = None,
                          frozen_receipts: Optional[Mapping[str, str]] = None,
                          root_inventory: Optional[Mapping[str, Any]] = None,
                          budget: Optional[SliceBudget] = None,
                          openai_client: Any = None,
                          shutdown: Optional[asyncio.Event] = None) -> Dict[str, Any]:
    """The ordinary incremental compaction (§1f default).

    The input span is NOT the lineage span: the crawl covers `(parent_boundary_ts, pinned_H]`
    while the resulting snapshot still claims the PARENT's `source_floor_ts` — its lineage covers
    everything the parent covered. Chunk zero is the parent's persisted payload bytes verbatim.
    """
    resolved = sizing or resolve_sizing(model=config.utility_model,
                                        window=config.get_model_token_limit(
                                            config.utility_model))
    frontier = await _max_observation_id(db, team_id, channel_id)
    checkpoint = new_checkpoint(
        team_id=team_id, channel_id=channel_id, namespace=namespace, pinned_H=h,
        mutation_frontier=frontier,
        source_floor_ts=str(parent.get("source_floor_ts")),
        input_floor_ts=str(parent.get("boundary_ts")), input_floor_inclusive=False,
        crawl_mode=CRAWL_MODE_INCREMENTAL, serializer_version=serializer_version,
        serializer_config_hash=serializer_config_hash, sizing=resolved,
        headroom_source=headroom_source, headroom_tokens=headroom_tokens,
        parent_snapshot_id=parent.get("snapshot_id"), frozen_renders=frozen_renders,
        frozen_receipts=frozen_receipts, root_inventory=root_inventory)
    await db.upsert_crawl_checkpoint_async(dict(checkpoint))
    return await run_crawl_slice(db=db, client=client, team_id=team_id, channel_id=channel_id,
                                 namespace=namespace, coordinator=coordinator,
                                 trigger="incremental", headroom_source=headroom_source,
                                 headroom_tokens=headroom_tokens, budget=budget,
                                 openai_client=openai_client, shutdown=shutdown,
                                 # THE PARENT IS THE ACTIVE POINTER, so it is what the publication
                                 # CAS must expect. Omitting this expects NO active pointer, and an
                                 # incremental generation would lose the CAS to its own parent and
                                 # physically delete the candidate it just spent a crawl building.
                                 expected_previous_id=parent.get("snapshot_id"),
                                 serializer_version=serializer_version, sizing=resolved)
    # `live` is deliberately NOT passed: this call has just written a FRESH checkpoint from
    # current configuration, so there is no prior progress for the step-2 ladder to compare
    # against. Every LATER slice arrives through the coordinator, which does pass it. Deriving
    # `live` from the checkpoint here would make each comparison a no-op by construction.
