"""The channel stream: fetch, discovery, pinning, serialization (spec §1–§4).

One channel turn renders TWO blocks. The PERIPHERY is a shallow recent window of the room — every
eligible event at or above the SELECTED floor `F'`, threads interleaved — and it IS a
last-N-shaped view, with replies riding along uncounted. Older history exists below it, reachable
by tool, and the horizon line says so. After the cache breakpoint comes the ORIGIN block: the
COMPLETE thread the turn was asked in, never truncated and never floored.

`F` AND `F'` ARE DIFFERENT VALUES AND THE DISTINCTION IS LOAD-BEARING. `F` is the floor READ from
`channel_window_anchor` at the start of the turn; `F'` is the floor this build SELECTED and the
only one that filters. They are usually equal — a window inside its bounds keeps its floor, which
is what makes the cached prefix survive — but when the roots above `F` exceed
`CHANNEL_WINDOW_CEILING`, `_select_floor` advances `F'` to the oldest of the newest
`CHANNEL_WINDOW_TARGET` roots and everything below it drops out of the render. `F'` is what is
persisted, what the horizon states, and what `stream_render.periphery_floor_ts` carries.

ONLY THE PRE-BREAKPOINT HALF IS ORIGIN-INDEPENDENT, and that is the invariant worth protecting:
one shared periphery pin serialized under two different origins must produce byte-identical
canonical items and the same `stream_sha256`, so every thread in a channel shares one cache
prefix. The origin block is origin-dependent by construction and sits below the breakpoint where
that costs nothing. The two-origin probe measures this against live data.

Pinning is STAGED, not all-before-Slack. `prepare_channel_turn` pins H, the frontier drain, the
serializer config and READ 1 (the anchor and the inventory) before any fetch exists. READ 2 — the
sidecar rows — comes AFTER the fetch, because its subject is the candidate identities the fetch
returned and eligibility cannot be decided without them. A retry reuses every pin rather than
re-reading the world, because a retry that re-read it would answer a different question than the
one that failed.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import (Any, Callable, Dict, FrozenSet, Iterable, List, Mapping,
                    NamedTuple, Optional, Sequence, Tuple)

from base_client import ChannelStreamError, HistoryFetchError
from config import config
from database import is_unattended_summary
from logger import setup_logger
import prompts
from message_processor import dev_barriers
from message_processor.turn_runtime import (AuthorizedEditTarget,
                                            RECEIPT_CLASS_ASSISTANT_REPLY)
from message_processor.utilities import api_part
from openai_client.base import attach_cache_breakpoint
from slack_client import actor_tail as actor_tail_module
from slack_client import admission_watermark
from slack_sdk.errors import SlackApiError

from slack_client.history_fetch import (FetchBudget, HistoryPageError, iter_pages,
                                        page_messages, slack_error_code)
from slack_client.utilities import ACTOR_REMOTE_LOOKUP_DEFAULT
from slack_client.normalizer import (
    NormalizedMessage,
    ORIGIN_HISTORY,
    ORIGIN_REPLIES,
    TimestampError,
    in_window,
    normalize_slack_message,
    parse_ts,
    render_mentions,
    sanitize_field,
    sanitize_name,
)

logger = setup_logger(name="slack_bot.ChannelStream")

# The reach-tool ORDER, shared with the prompt guidance so the horizon and the
# instructions can never name the same set in two different orders.
REACH_TOOLS = prompts.REACH_TOOLS

# v4: the pinned authorization facts changed — `ReceiptRec` gained `receipt_class` and the
# message-item metadata gained `edited_ts`, the two facts the edit_own_message authorization
# mapping (EDIT §2a) is built from. Rendered bytes are unchanged; the pin's contents are not.
# v5: one marker's bytes changed — a reaction the bot itself placed renders a " (you)"
# suffix in the reactions marker (render_reactions_marker), carried by ReactionRec.mine.
SERIALIZER_VERSION = 5

# Versions the SELECTION POLICY — floor semantics, target/ceiling arithmetic,
# eligibility — separately from the serializer GRAMMAR, because a policy change must
# invalidate a persisted floor without necessarily changing a rendered byte.
SELECTION_VERSION = 1

# ---------------------------------------------------------------- grammar constants
# Every one of these is part of the serialized bytes, so every one is a version-pinned
# constant rather than a call-site literal. Changing any of them changes SERIALIZER_VERSION.
# The horizon carries only SLOW-MOVING facts. H is deliberately absent: this is item 0 of the
# cacheable prefix, and a per-turn value here would invalidate the whole stream beneath it on
# every turn. The live edge is stated post-breakpoint in the turn coordinates and recorded in
# `stream_render`.
# v5 changed one marker's bytes (a reaction the bot placed renders " (you)"); everything
# else in this block is v3's, unchanged.
HORIZON_AWARENESS_LINE = (
    "Images and code-execution results in this stream are awareness-only outside the current "
    "thread; current-thread file, image and container details follow after the stream."
)
HORIZON_TEMPLATE = (
    "[STREAM HORIZON: the recent activity in this channel, from {floor_ts}"
    "{reach_clause}{index_clause}]\n"
    + HORIZON_AWARENESS_LINE
)
# A1's no-floor variant: the floor clause is omitted ENTIRELY rather than naming a ts. Reached
# when the periphery holds no eligible events at all — there is then no message whose ts the
# window could honestly claim to begin at.
HORIZON_TEMPLATE_NO_FLOOR = (
    "[STREAM HORIZON: no recent messages in this channel{reach_clause}{index_clause}]\n"
    + HORIZON_AWARENESS_LINE
)
# The reach segment is OMITTED ENTIRELY when no reach tool is exposed — the model is never told
# history is reachable by a tool it cannot call.
HORIZON_REACH_TEMPLATE = "; older history exists and is reachable with {reach_list}"

# A2. The post-breakpoint origin header. Its second line is what stops the model reading the
# deliberate duplication — an origin message inside the window renders in BOTH blocks — as two
# separate exchanges. Load-bearing; do not trim it.
ORIGIN_HEADER_TEMPLATE = (
    "[CURRENT THREAD — thread={origin_root_ts}, complete, {origin_count} messages]\n"
    "This is the whole of the thread you are being asked in. Messages from it may also appear "
    "above, in the channel's recent activity; that is the same thread seen from the room."
)

# A5. The orphaned-reply marker. A reply at or above the floor may belong to a root BELOW it;
# that reply renders a `thread=<root_ts>` label pointing at a root the model cannot see anywhere,
# which is precisely the shape that invites invention.
ORPHAN_MARKER_TEMPLATE = "[thread={root_ts} began before this window{tool_clause}]"
# Named ONLY when the tool is actually exposed — history tools are a global switch, and with
# them off this would instruct the model to call something it does not have.
ORPHAN_MARKER_TOOL_CLAUSE = " — use fetch_thread_messages to read it"

# The §2f index clause, one per InventoryPin state. `warm` renders nothing: an index that reaches
# everything has no caveat to state, and a sentence saying so would be noise in every horizon.
# The clauses describe THE THREAD INDEX only — never the stream's own reach, which the floor above
# already names.
INDEX_CLAUSE_UNINDEXED = ("; I have not indexed this channel's older threads yet, so a recent "
                          "reply under an older thread may be missing")
INDEX_CLAUSE_RETENTION = "; Slack's retention limits how far back my thread index reaches"
INDEX_CLAUSE_DEPTH_TEMPLATE = "; my thread index reaches back {days} days"
INDEX_CLAUSE_UNAVAILABLE = "; I could not index this channel's older threads"

# §2f's SIX states, and the list is closed. `pending` and `running` both map to `cold`, which is
# why seven rows yield six states.
INVENTORY_ABSENT = "absent"
INVENTORY_COLD = "cold"
INVENTORY_WARM = "warm"
INVENTORY_LIMITED_RETENTION = "limited_retention"
INVENTORY_LIMITED_DEPTH = "limited_depth"
INVENTORY_UNAVAILABLE = "unavailable"
INVENTORY_STATES = (INVENTORY_ABSENT, INVENTORY_COLD, INVENTORY_WARM,
                    INVENTORY_LIMITED_RETENTION, INVENTORY_LIMITED_DEPTH, INVENTORY_UNAVAILABLE)

REASON_UNAVAILABLE = "unavailable"

END_MARKER_TEXT = "[end of channel stream]"

# A7. Any payload LINE beginning with one of these is prefixed with "· " before it is persisted
# and again before it is rendered, so nothing a model wrote can present itself as our structure.
A2_RESERVED_PREFIXES: Tuple[str, ...] = (
    "[CURRENT THREAD",
    "[thread=",
    "[CHANNEL SUMMARY",
    "[ROOT ANCHORS",
    "[END CHANNEL SUMMARY",
    "[STREAM HORIZON",
    "[EARLIER ARTIFACT",
    "[THIS THREAD BEFORE",
    "[END EARLIER THREAD CONTEXT",
    "[end of channel stream]",
    "[NOTE:",
)
ESCAPE_PREFIX = "· "
SNIPPET_WORDS = 6
SNIPPET_CHARS = 48
DELETED_SNIPPET = '~"[deleted]"'
IMAGE_GIST_CHARS = 200
AMBIENT_NOTE_CHARS = 400
REACTIONS_RENDERED = 2
FILES_MARKER_LIMIT = 10

MARKER_KIND_AMBIENT = "ambient"
MARKER_KIND_DOCUMENT = "document"
MARKER_KIND_IMAGE = "image_analysis"
MARKER_KIND_TOOL = "tool_provenance"

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

RECEIPT_FINALIZED = "finalized"


# ---------------------------------------------------------------- exceptions

# ChannelStreamError is re-exported from base_client, where HistoryFetchError also lives —
# one hierarchy, rooted below this module so a channel turn can catch every fail-closed
# context failure with one clause.


class OriginFetchError(ChannelStreamError):
    """The ORIGIN thread could not be read completely. Never answer from a partial origin.

    NAMED, NOT STRING-MATCHED, and it CARRIES THE SLACK CODE — without that the §2e taxonomy is
    unusable, because wrapping destroys the very string it branches on. `_channel_stream_failure`
    branches on the TYPE and returns `origin_fetch_failed`; the periphery's own failures stay
    `HistoryFetchError` and return `history_fetch_failed`.
    """

    def __init__(self, message: str, *, code: Optional[str] = None):
        super().__init__(message)
        # "" / None when the cause carried no Slack code. `None` is NOT a taxonomy match and
        # therefore FAILS CLOSED, which is the allowlist's default.
        self.code = code


class SidecarPinMismatch(ChannelStreamError):
    """The two halves of the render pin describe two different worlds.

    `receipt_feature_epoch_ts` is a property of the CHANNEL, not of an id list, so two reads in
    one turn against one database must return the same value. A difference means something is
    racing the feature-epoch write, and rendering half the stream under one epoch and half under
    another would produce bytes no single read of the database supports.
    """


class FreezeError(ChannelStreamError):
    """§1p: a pin carries a cycle, or a value of a type the freeze policy has no rule for."""


class StreamOverBudgetError(ChannelStreamError):
    """The assembled request cannot fit the model's context even before the API sees it."""


class StreamTimestampError(ChannelStreamError, ValueError):
    """A timestamp on the turn path could not be parsed, so the turn fails closed.

    A malformed ts cannot be placed in the window, which means we cannot say whether the record
    belongs in the stream — and a record silently dropped for that reason is exactly the invisible
    hole the single stream exists to rule out. (An unsupported SUBTYPE is different: it is a
    deliberate "this is not a message" and still normalizes to None.)

    Both bases are load-bearing: it IS a bad value, so `except ValueError` call sites that predate
    the channel stream still catch it, and it IS a fail-closed context failure, so base.py's
    ChannelStreamError branch turns it into an honest notice rather than "something went wrong".
    """


def _checked_ts(value: Any, what: str) -> str:
    """A timestamp we are about to pin. Malformed or missing ⇒ the turn fails closed."""
    try:
        parse_ts(value)
    except TimestampError as e:
        raise StreamTimestampError(f"{what}: {e}") from e
    return str(value)


_FREEZE_SCALARS = (str, bytes, bytearray, bool, int, float, complex, type(None))


def freeze_deep(value: Any) -> Any:
    """§1p: the ONE recursive freeze, for sidecar rows and item metadata.

    Mappings become read-only proxies over frozen copies, sequences become tuples, sets become
    frozensets, scalars pass through. A CYCLE and an unsupported type are both BUILD ERRORS: a
    pin that cannot be frozen is a pin that can change after its bytes were hashed, and guessing
    at a type's mutability is how one slips through.
    """
    return _freeze(value, frozenset())


def _freeze(value: Any, path: frozenset) -> Any:
    if isinstance(value, _FREEZE_SCALARS):
        return bytes(value) if isinstance(value, bytearray) else value
    if id(value) in path:
        raise FreezeError(
            f"a {type(value).__name__} in this pin refers to itself; a cyclic pin cannot be "
            "frozen and must not be rendered")
    path = path | {id(value)}
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item, path) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, path) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item, path) for item in value)
    raise FreezeError(f"cannot freeze a value of type {type(value).__name__}")


def _frozen_row(row: Any) -> Mapping[str, Any]:
    """A sidecar row as a deeply read-only mapping. The renderers read rows with `.get`, so a
    proxy is all they need — and a proxy is what stops a caller mutating a row after its bytes
    were hashed."""
    if isinstance(row, MappingProxyType):
        return row
    return freeze_deep(dict(row or {}))


def _frozen_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({
        key: tuple(value) if isinstance(value, list) else value
        for key, value in payload.items()})


def _frozen_config_deep(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """`_frozen_config`, but recursively — a mapping member would otherwise stay mutable behind
    a read-only proxy, which is a freeze that only looks like one."""
    return freeze_deep(dict(payload))


# ---------------------------------------------------------------- pinned records

@dataclass(frozen=True)
class InventoryPin:
    """The thread-index row as the stream sees it: where the index starts, and §2f's ONE state.

    `state` is DERIVED from `(status, reason)` rather than stored, so the six-value vocabulary
    has exactly one definition and a consumer cannot re-derive it differently.
    """
    start_ts: str
    status: str
    reason: Optional[str]

    @property
    def state(self) -> str:
        status = str(self.status or "")
        if status == "complete":
            return INVENTORY_WARM
        if status in ("pending", "running"):
            return INVENTORY_COLD
        if status != "limited":
            return INVENTORY_COLD
        reason = str(self.reason or "")
        if reason == REASON_UNAVAILABLE:
            return INVENTORY_UNAVAILABLE
        if reason == "depth_config":
            return INVENTORY_LIMITED_DEPTH
        # NULL or unrecognised reads as retention — see index_clause's fail-safe note.
        return INVENTORY_LIMITED_RETENTION


@dataclass(frozen=True)
class ReceiptRec:
    ts: str
    state: str
    turn_id: Optional[str]
    thread_root_ts: Optional[str]
    # EDIT §2a/§4. The receipt's class, or None for a legacy row — and a legacy row authorizes
    # NO edit, by comparison against `RECEIPT_CLASS_ASSISTANT_REPLY` rather than by inference.
    receipt_class: Optional[str] = None


@dataclass(frozen=True)
class SidecarPin:
    """Every DB row the stream renders from.

    READ 2 FOLLOWS THE FETCH, and its subject is the CANDIDATE identities the fetch returned:
    eligibility needs receipt state and the receipt epoch, so a pin restricted to already-selected
    ids could never be built at all. Discovery moved to READ 1 and reads its own rows.

    IT IS ONE TRANSACTION PER CALL, AND THE BUILDER MAKES TWO. READ 2a covers the periphery
    candidate ids in the shared phase; READ 2b covers the origin-only ids per origin; and
    `merge_sidecar_pins` combines them with the SHARED rows winning every overlap, so the
    periphery's bytes are fixed before any origin is known. Each call is still one transaction
    over an exact id list, which is the property that makes a pin a pin.

    Deeply immutable, and validated here rather than at first use. `__post_init__` is freeze time:
    the rows become read-only mappings (a frozen dataclass holding mutable dicts is a pin that can
    change after its hash was recorded) and every timestamp is parsed once. A row whose ts cannot
    be parsed fails the turn HERE, before it can be silently skipped by whichever consumer reaches
    it first.
    """
    window: Tuple[str, bool]
    receipts: Tuple[ReceiptRec, ...]
    receipt_feature_epoch_ts: Optional[str]
    coverage: Optional[InventoryPin]
    activity_roots: Tuple[str, ...]
    activity_event_ts: Tuple[Tuple[str, Optional[str]], ...]
    image_analyses: Tuple[Mapping[str, Any], ...]
    document_extractions: Tuple[Mapping[str, Any], ...]
    ambient_artifacts: Tuple[Mapping[str, Any], ...]
    tool_usage: Tuple[Tuple[str, Tuple[Mapping[str, Any], ...]], ...]
    versions_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "window",
                           (_checked_ts(self.window[0], "window floor"), bool(self.window[1])))
        if self.coverage is not None:
            _checked_ts(self.coverage.start_ts, "inventory_start_ts")
        for rec in self.receipts:
            _checked_ts(rec.ts, "receipt message_ts")
            if rec.thread_root_ts:
                _checked_ts(rec.thread_root_ts, "receipt thread_root_ts")
        for root in self.activity_roots:
            _checked_ts(root, "activity root_ts")
        for root, event_ts in self.activity_event_ts:
            _checked_ts(root, "activity root_ts")
            if event_ts:
                _checked_ts(event_ts, "activity last_index_event_ts")
        object.__setattr__(self, "image_analyses",
                           self._rows(self.image_analyses, "message_ts", "image analysis"))
        object.__setattr__(self, "document_extractions",
                           self._rows(self.document_extractions, "message_ts", "document"))
        object.__setattr__(self, "ambient_artifacts",
                           self._rows(self.ambient_artifacts, "source_ts", "ambient artifact"))
        object.__setattr__(self, "tool_usage", tuple(
            (_checked_ts(ts, "tool usage message_ts"), tuple(_frozen_row(t) for t in tools))
            for ts, tools in self.tool_usage))

    @staticmethod
    def _rows(rows: Sequence[Any], ts_key: str,
              what: str) -> Tuple[Mapping[str, Any], ...]:
        frozen = []
        for row in rows:
            row = _frozen_row(row)
            _checked_ts(row.get(ts_key), f"{what} {ts_key}")
            frozen.append(row)
        return tuple(frozen)

    def receipt_for(self, ts: str) -> Optional[ReceiptRec]:
        for rec in self.receipts:
            if rec.ts == ts:
                return rec
        return None


@dataclass(frozen=True)
class PinnedTuple:
    """Everything the build is a pure function of, once the world has been read.

    Evidence and config that vary with origin or requester live OUTSIDE this tuple, in the
    assembler's own evidence snapshot — otherwise two requesters in one channel could never
    share a cache prefix, which is the thing the single stream exists to make possible.

    `chrome_ts` is SUPPLIED and KEPT: the builder classified it as pages arrived, selection
    already acted on that value, and `__post_init__` recomputes it only to VALIDATE — raising
    on a mismatch rather than substituting its own answer, which would silently discard what
    selection used and reintroduce the divergence pinning the candidates exists to prevent.

    `sidecar_markers` is DERIVED at pin time rather than passed in, and that is the point: it is
    the output of a renderer that reads live configuration (the provenance/ambient annotation
    budgets). Rendering it here, once, means the serializer reads nothing but this tuple — so the
    same pin re-serialized after a config change still produces the bytes the turn was admitted
    with, and a config change is a cache MISS via `serializer_config_hash` rather than a silently
    different stream under the same hash. Passing it in is ignored: it is a function of the other
    fields and is always computed here, so there is no way to hand the serializer markers its
    rows do not support.

    `chrome_ts` IS THE EXCEPTION AND IT IS NOT IGNORED. It arrives from the builder, which
    classified it as pages arrived and which SELECTION already acted on; `__post_init__`
    recomputes it only to compare. The two acts differ on purpose — recompute-and-compare proves
    the supplied value honest, while recompute-and-USE would discard what selection acted on and
    reintroduce the divergence between selection and rendering that pinning the candidates exists
    to prevent.
    """
    team_id: str
    channel_id: str
    window: Tuple[str, bool]
    H: str
    fetch_snapshot: Tuple[NormalizedMessage, ...]          # PERIPHERY candidates
    sidecar_versions_hash: str
    actor_map: Tuple[Tuple[str, str], ...]
    actor_map_hash: str
    serializer_version: int
    serializer_config_hash: str
    capability_profile_hash: str
    tool_schema_version: str
    coverage: Optional[InventoryPin]
    receipt_feature_epoch_ts: Optional[str]
    receipt_map: Tuple[Tuple[str, str, Optional[str], Optional[str]], ...]
    sidecars: SidecarPin
    # --- the shallow window's own pins ------------------------------------------------------
    origin_root_ts: Optional[str] = None
    origin_snapshot: Tuple[NormalizedMessage, ...] = ()    # ORIGIN candidates, complete
    periphery_floor_ts: str = ""                           # F' — "" is the empty-floor sentinel
    selection_version: int = 0
    reach_tools: Tuple[str, ...] = ()
    serializer_config: Mapping[str, Any] = field(default_factory=dict)
    sidecar_markers: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    chrome_ts: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        _checked_ts(self.H, "H")
        # The empty-floor sentinel is NOT a timestamp and `_checked_ts` must never see it —
        # `parse_ts("")` raises, and "this channel has no eligible events" is a legal state.
        if self.window[0]:
            _checked_ts(self.window[0], "window floor")
        object.__setattr__(self, "window", (str(self.window[0]), bool(self.window[1])))
        object.__setattr__(self, "origin_snapshot", tuple(self.origin_snapshot))
        object.__setattr__(self, "reach_tools", tuple(self.reach_tools))
        config_values = _frozen_config(self.serializer_config or serializer_config_snapshot())
        object.__setattr__(self, "serializer_config", config_values)
        object.__setattr__(self, "sidecar_markers", _render_sidecar_markers(
            self.sidecars, self.channel_id, config_values))

        # ONE OPERATIONAL CLASSIFICATION PLUS ONE VALIDATION RECOMPUTATION. The builder classified
        # chrome as pages arrived — the walk's stopping predicate needs it — and that SUPPLIED
        # value is what selection already acted on and what the serializer renders. Recomputing
        # and USING the result here would silently discard it, reintroducing the divergence
        # between selection and rendering that pinning the candidates exists to prevent.
        #
        # The subject is the DEDUPED UNION of both snapshots: an origin-only message is subject to
        # the same eligibility rule, so validating over the periphery alone would leave exactly
        # the origin-only chrome unchecked — the own-output exclusion rule defeated by a gap in
        # coverage rather than by a bad predicate.
        supplied = frozenset(self.chrome_ts)
        recomputed = classify_chrome(
            _dedup(tuple(self.fetch_snapshot) + tuple(self.origin_snapshot)),
            chrome_markers=config_values.get("chrome_markers", ()))
        if supplied != recomputed:
            raise FreezeError(
                "the pinned chrome classification does not match a recomputation over the same "
                f"messages and markers (supplied-only={sorted(supplied - recomputed)}, "
                f"recomputed-only={sorted(recomputed - supplied)}); the pin and the selection "
                "that acted on it disagree about which of our own messages are chrome")
        object.__setattr__(self, "chrome_ts", supplied)

    @property
    def inventory_state(self) -> str:
        """§2f's state, with `absent` covering the channel that has no row at all."""
        return self.coverage.state if self.coverage is not None else INVENTORY_ABSENT

    @property
    def horizon_floor_ts(self) -> str:
        """WHERE THE RENDERED WINDOW ACTUALLY STARTS, which is what the horizon may claim.

        It is the SELECTED periphery floor `F'` — the policy decision this build made, not the
        inventory's reach and not the oldest message that happened to be fetched. `""` is the
        empty-floor sentinel and selects A1's no-floor variant; `parse_ts` is never called on it.
        """
        return self.periphery_floor_ts

    @property
    def floor_ts(self) -> str:
        return self.window[0]

    @property
    def floor_inclusive(self) -> bool:
        return self.window[1]

    @property
    def actor_names(self) -> Dict[str, str]:
        return dict(self.actor_map)

    def marker_lines(self, ts: str) -> Tuple[str, ...]:
        for source_ts, lines in self.sidecar_markers:
            if source_ts == ts:
                return lines
        return ()


class SharedPageCounts(NamedTuple):
    history: int
    reply: int


class PageCounts(NamedTuple):
    history: int                  # each read from its own FetchBudget.pages_used
    reply: int
    origin: int


@dataclass(frozen=True)
class PreparedTurn:
    """§4.4 steps 1-4: everything that must complete BEFORE any fetch starts.

    It exists as its own awaited phase so the ordering is STRUCTURAL rather than merely
    discouraged: the origin fetch runs concurrently with the periphery build, so if the drain
    lived inside that build, Slack I/O could begin before the admission watermark had resolved —
    the one ordering P1 established. There is no code path from the composer to a fetch that does
    not first await this.
    """
    team_id: str
    channel_id: str
    h: str                                  # already through _checked_ts
    frontier: int                           # the drained frontier
    floor_read: Optional[str]               # F as READ (None when UNSET or a stale version)
    coverage: Optional[InventoryPin]        # READ 1 stage 1's inventory; None means ABSENT
    generation: int                         # actor-tail generation, captured BEFORE the fetch
    selection_version: int
    serializer_config: Mapping[str, Any]    # frozen at step 2a


@dataclass(frozen=True)
class OriginFetch:
    """The result of ONE origin fetch, before any pin."""
    origin_root_ts: Optional[str]
    messages: Tuple[NormalizedMessage, ...]   # normalized, deduped, EVERY fetched message
    pages: int                                # from its own FetchBudget.pages_used
    empty_fallback: bool                      # §2e's ONE legal empty origin was taken
    deadline_at: Optional[float]              # THE BUDGET'S deadline, carried forward so the pin
                                              # that consumes it can prove one shared clock


@dataclass(frozen=True)
class StreamBuildResult:
    """What the BUILD produced, as opposed to what the pin contains.

    `reselected` and `anchor_advanced` are not on `ChannelStream` because neither is knowable at
    serialization: `reselected` is a property of the SELECTION, and `anchor_advanced` is the
    UPSERT's return value, which does not exist until after the bytes are already frozen.
    """
    stream: ChannelStream
    reselected: bool              # this build chose a floor different from the one it read
    anchor_advanced: bool         # the UPSERT moved the row; False on skip AND on failure
    pages: PageCounts


def merge_sidecar_pins(shared: SidecarPin, origin: SidecarPin, *,
                       ids: Sequence[str]) -> SidecarPin:
    """READ 2a + READ 2b -> the ONE pin the serializer renders from.

    `ids` is the SORTED UNION of what the two reads asked for. It is a PARAMETER because
    `SidecarPin` has no id field: the id list enters the HASH, not the pin's shape.

    THE ONE SCALAR MUST AGREE. `receipt_feature_epoch_ts` is a property of the CHANNEL, not of an
    id list, so two reads in one turn against one database must return the same value. A
    difference means something is racing the feature-epoch write and the two halves describe two
    different worlds — it RAISES and the turn fails closed, because rendering half the stream
    under one epoch and half under another produces bytes no single read supports.

    SHARED WINS PER GROUP, AND A GROUP IS EVERY ROW FOR ONE SUBJECT TS. `image_analyses`,
    `document_extractions` and `ambient_artifacts` are NOT unique by timestamp — one message can
    carry three images, and the accessor selects the row `id` and orders by `(ts, id)` precisely
    because of it. Keying those lists by ts and merging `{**origin, **shared}` would collapse each
    group to ONE row and silently drop the rest: an artifact marker vanishing from the render.

    `window` and `coverage` are NOT merged and cannot diverge — both READ-2 pins carry the inert
    values, and nothing reads them. The floor that renders lives on `PinnedTuple.window`; the
    inventory that renders lives on `PinnedTuple.coverage`.
    """
    if shared.receipt_feature_epoch_ts != origin.receipt_feature_epoch_ts:
        raise SidecarPinMismatch(
            "the two halves of the render pin disagree about the receipt feature epoch "
            f"({shared.receipt_feature_epoch_ts!r} vs {origin.receipt_feature_epoch_ts!r}); "
            "something is racing the epoch write and the pin describes two different worlds")

    def _grouped(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, List[Mapping]]:
        out: Dict[str, List[Mapping]] = {}
        for row in rows:
            out.setdefault(str(row.get(key)), []).append(row)
        return out

    def _merge_groups(shared_rows, origin_rows, key: str) -> Tuple[Mapping[str, Any], ...]:
        groups = _grouped(origin_rows, key)
        # THE SHARED GROUP WINS WHOLE — every row in it, not one row from it. The two id sets are
        # disjoint by construction, so an overlap means a row came back for an id nobody asked
        # for; resolving it to the shared group is right because a duplicate row is not the
        # evidence of a race that a divergent scalar is.
        groups.update(_grouped(shared_rows, key))
        merged = [row for rows in groups.values() for row in rows]
        # The accessor's OWN order, so a merged pin is byte-comparable with a single read's.
        return tuple(sorted(merged, key=lambda r: (_ts_or_zero(r.get(key)),
                                                   r.get("id") or 0)))

    receipts = {rec.ts: rec for rec in origin.receipts}
    receipts.update({rec.ts: rec for rec in shared.receipts})
    tool_usage = dict(origin.tool_usage)
    tool_usage.update(dict(shared.tool_usage))

    return SidecarPin(
        window=shared.window,
        receipts=tuple(sorted(receipts.values(), key=lambda r: parse_ts(r.ts))),
        receipt_feature_epoch_ts=shared.receipt_feature_epoch_ts,
        coverage=shared.coverage,
        activity_roots=(),
        activity_event_ts=(),
        image_analyses=_merge_groups(shared.image_analyses, origin.image_analyses,
                                     "message_ts"),
        document_extractions=_merge_groups(shared.document_extractions,
                                           origin.document_extractions, "message_ts"),
        ambient_artifacts=_merge_groups(shared.ambient_artifacts, origin.ambient_artifacts,
                                        "source_ts"),
        tool_usage=tuple(sorted(tool_usage.items(), key=lambda pair: parse_ts(pair[0]))),
        # RECOMPUTED over the merged rows plus `ids` by the same function the accessor uses —
        # never a hash of the two hashes, which would move whenever the periphery/origin split
        # moved even though the rendered rows were identical, and would break the cache for
        # nothing.
        versions_hash=_merged_versions_hash(
            receipts=receipts, tool_usage=tool_usage, ids=ids,
            epoch=shared.receipt_feature_epoch_ts,
            images=_merge_groups(shared.image_analyses, origin.image_analyses, "message_ts"),
            documents=_merge_groups(shared.document_extractions, origin.document_extractions,
                                    "message_ts"),
            ambient=_merge_groups(shared.ambient_artifacts, origin.ambient_artifacts,
                                  "source_ts")),
    )


def _ts_or_zero(raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _merged_versions_hash(*, receipts, tool_usage, ids, epoch, images, documents,
                          ambient) -> str:
    """The accessor's own hash, over merged material. Delegated so the merge and a single read
    cannot produce different hashes for the same rendered rows."""
    from database import DatabaseManager
    return DatabaseManager._sidecar_versions_hash({
        "ids": list(ids),
        "receipt_feature_epoch_ts": epoch,
        "receipts": [{"message_ts": r.ts, "state": r.state, "turn_id": r.turn_id,
                      "thread_root_ts": r.thread_root_ts, "receipt_class": r.receipt_class}
                     for r in sorted(receipts.values(), key=lambda r: parse_ts(r.ts))],
        "image_analyses": [dict(r) for r in images],
        "document_extractions": [dict(r) for r in documents],
        "ambient_artifacts": [dict(r) for r in ambient],
        "tool_usage": {ts: [dict(t) for t in tools] for ts, tools in tool_usage.items()},
    })


@dataclass(frozen=True)
class SharedChannelPin:
    """Everything an ORIGIN-INDEPENDENT build produces — fetched and read ONCE per turn.

    This is what makes one periphery serializable under N origins: the probe builds one of these
    and two `PinnedTuple`s from it, and the two must render byte-identical pre-breakpoint bytes.
    """
    team_id: str
    channel_id: str
    h: str
    deadline_at: float                               # the ONE shared deadline
    serializer_config: Mapping[str, Any]
    generation: int
    floor_read: Optional[str]                        # F as READ (None when UNSET)
    periphery_floor_ts: str                          # F' — "" is the empty-floor sentinel
    reselected: bool
    periphery_candidates: Tuple[NormalizedMessage, ...]
    periphery: Tuple[NormalizedMessage, ...]         # the SELECTION: eligible, at/above F'
    periphery_sidecars: SidecarPin                   # READ 2a, over PERIPHERY candidate ids
    periphery_chrome_ts: FrozenSet[str]
    actor_map: Tuple[Tuple[str, str], ...]           # THE PERIPHERY ACTOR MAP, frozen here
    actor_ids_attempted: FrozenSet[str]              # every id the periphery TRIED to resolve
    actor_lookups_remaining: int                     # remote budget left for origin-only actors
    coverage: Optional[InventoryPin]                 # None means ABSENT
    selection_version: int
    reach_tools: Tuple[str, ...]
    capability_profile_hash: str
    tool_schema_version: str
    pages: SharedPageCounts

    def __post_init__(self) -> None:
        """DEEP-FREEZE. `frozen=True` protects the ATTRIBUTE, not what it POINTS AT, and
        `serializer_config_snapshot()` returns a mutable dict whose `chrome_markers` is a mutable
        list. Without this, code between two `build_origin_pin` calls could mutate the marker
        list and fork the very prefix those two calls exist to prove identical.
        """
        object.__setattr__(self, "serializer_config", _frozen_config_deep(
            self.serializer_config))
        object.__setattr__(self, "periphery_candidates", tuple(self.periphery_candidates))
        object.__setattr__(self, "periphery", tuple(self.periphery))
        object.__setattr__(self, "periphery_chrome_ts", frozenset(self.periphery_chrome_ts))
        object.__setattr__(self, "actor_map", tuple(tuple(pair) for pair in self.actor_map))
        object.__setattr__(self, "actor_ids_attempted", frozenset(self.actor_ids_attempted))
        object.__setattr__(self, "reach_tools", tuple(self.reach_tools))

    @property
    def root_count(self) -> int:
        """DISTINCT TOP-LEVEL ROOTS IN THE SELECTION — §2d's count subject.

        Derived, never stored: it is a function of `periphery`, and a stored copy could disagree
        with it. Equals `ChannelStream.root_count` by construction, because the serializer renders
        from a pin carrying the same floor over the same candidates.
        """
        return len({m.ts for m in self.periphery if not m.is_reply})


@dataclass(frozen=True)
class StreamItem:
    role: str
    content: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _frozen_row(self.metadata))


@dataclass(frozen=True)
class ChannelStream:
    pinned: PinnedTuple
    items: Tuple[StreamItem, ...]          # CANONICAL, pre-breakpoint: horizon + periphery
    horizon_item: StreamItem
    end_marker_item: StreamItem
    message_items: Tuple[StreamItem, ...]  # the periphery MESSAGES only, markers excluded
    origin_header_item: Optional[StreamItem]   # None only when the origin block is empty
    origin_items: Tuple[StreamItem, ...]       # the COMPLETE origin thread, post-breakpoint
    stream_sha256: str                     # over `items` — THE cross-origin equality value
    union_sha256: str                      # `items`, then the origin header, then origin items
    byte_count: int                        # ALL pre-breakpoint item content, markers included
    origin_byte_count: int                 # the origin header + origin items
    message_count: int                     # rendered periphery MESSAGE items only
    origin_count: int                      # rendered origin MESSAGE items only, header excluded
    candidate_count: int                   # periphery candidates BEFORE eligibility and the count
    root_count: int                        # FINAL RENDERED roots, after the floor filter
    orphan_root_count: int                 # A5 marker ITEMS rendered
    periphery_floor_ts: str                # "" is the empty-floor sentinel
    selection_version: int
    inventory_state: str                   # one of §2f's SIX values
    serializer_version: int
    serializer_config_hash: str
    sidecar_versions_hash: str
    actor_map_hash: str
    capability_profile_hash: str
    receipts_included: Tuple[str, ...]
    receipts_excluded: Tuple[str, ...]
    receipts_membership_hash: str

    # -- post-serialization selectors ------------------------------------------------------

    def origin_slice(self, origin_thread_ts: Optional[str]) -> Tuple[StreamItem, ...]:
        """The origin thread's items. RETAINED BY NAME AND SIGNATURE so existing callers keep
        working, but ITS MEANING CHANGED: the origin is now FETCHED rather than selected out of
        the window, so this returns `origin_items` when the argument matches the origin this
        build was made for, and `()` otherwise."""
        if not origin_thread_ts:
            return ()
        if self.pinned.origin_root_ts and str(origin_thread_ts) == str(
                self.pinned.origin_root_ts):
            return self.origin_items
        return ()

    def trigger_view(self, trigger_ts: Optional[str]) -> Optional[StreamItem]:
        """The item for one ts, or None when Slack had not propagated it by the time we
        fetched — the caller then falls back to the verbatim trigger block."""
        if not trigger_ts:
            return None
        for item in (*self.message_items, *self.origin_items):
            if item.metadata.get("ts") == str(trigger_ts):
                return item
        return None

    @property
    def trusted_thread_roots(self) -> frozenset:
        """The thread roots this stream actually LABELLED for the model.

        The allowlist `post_to_thread` authorizes a cross-thread target against, and it is read
        off the SERIALIZED items rather than the fetch snapshot on purpose: the snapshot still
        carries the messages the serializer excluded — chrome, another turn's in-flight surface —
        and a target the model was never shown is a target it is guessing at.

        It is the set of `thread=<ts>` labels the rendered headers carry, PLUS the origin root.
        ORPHAN ROOTS ARE INCLUDED — a rendered `thread=<ts>` label is a rendered label whether or
        not its root is inside the window, and the marker above it tells the model how to read
        the thread before answering in it. A top-level message with no replies is still NOT in
        it: no thread label was rendered for it, and the schema promises only the labels.
        """
        labelled = {str(root) for root in
                    (item.metadata.get("thread_root_ts")
                     for item in (*self.message_items, *self.origin_items)) if root}
        if self.pinned.origin_root_ts:
            labelled.add(str(self.pinned.origin_root_ts))
        return frozenset(labelled)

    @property
    def authorized_edit_targets(self) -> Mapping[str, AuthorizedEditTarget]:
        """EDIT §2a. The exact own messages this stream SHOWED the model that are editable.

        Built from the serialized `message_items` + `origin_items` — what was rendered, never the
        fetch snapshot — deduplicated by ts, and a ts is a key only when ALL of these hold:

        * the item's role is `assistant` — a role one of our own messages gets ONLY through
          receipt-based resolution, so the role itself is receipt-vouched;
        * the durable receipt is `finalized`;
        * the receipt's class is `assistant_reply` — a legacy NULL-class row, chrome, a
          correction announcement and every other class fail here, by comparison and never by
          inference;
        * the item's channel is the pinned channel.

        `thread_root_ts` is the RECEIPT's, not the header's: it is the value the disclosure
        lands under (`receipt.thread_root_ts or message_ts`), and taking it from one source in
        both §2a and §2b is what lets duplicate appearances agree. `edited_ts` is the rendered
        metadata's snapshot. Duplicate appearances that disagree EXCLUDE the ts.
        """
        out: Dict[str, AuthorizedEditTarget] = {}
        conflicted: set = set()
        channel = self.pinned.channel_id
        for item in (*self.message_items, *self.origin_items):
            if item.role != ROLE_ASSISTANT:
                continue
            meta = item.metadata
            ts = meta.get("ts")
            if not ts or meta.get("channel_id") != channel:
                continue
            ts = str(ts)
            receipt = self.pinned.sidecars.receipt_for(ts)
            if receipt is None or receipt.state != RECEIPT_FINALIZED:
                continue
            if receipt.receipt_class != RECEIPT_CLASS_ASSISTANT_REPLY:
                continue
            target = AuthorizedEditTarget(
                channel_id=str(channel), message_ts=ts,
                thread_root_ts=receipt.thread_root_ts,
                edited_ts=meta.get("edited_ts"),
                receipt_class=str(receipt.receipt_class))
            held = out.get(ts)
            if held is not None and held != target:
                conflicted.add(ts)
                continue
            out[ts] = target
        for ts in conflicted:
            out.pop(ts, None)
        return MappingProxyType(out)

    def normalized_for(self, ts: Optional[str]) -> Optional[NormalizedMessage]:
        if not ts:
            return None
        for message in (*self.pinned.fetch_snapshot, *self.pinned.origin_snapshot):
            if message.ts == str(ts):
                return message
        return None

    # -- assembly ---------------------------------------------------------------------------

    def input_items(self) -> List[Dict[str, Any]]:
        """The stream as API input items, metadata stripped. The end marker's cache breakpoint
        is NOT attached here — see end_marker_content."""
        return [{"role": item.role, "content": item.content} for item in self.items]

    def end_marker_content(self, model: Optional[str]) -> List[Dict[str, Any]]:
        """The end marker as a content part carrying the explicit cache breakpoint.

        The breakpoint is attached AFTER api_part, because api_part STRIPS it — the P1 trap.
        The stream's whole cacheable prefix ends here, so losing this marker silently costs
        every channel turn its cache hit and nothing fails loudly.
        """
        part = api_part({"type": "input_text", "text": END_MARKER_TEXT})
        return [attach_cache_breakpoint(part, model)]

    def origin_input_items(self) -> List[Dict[str, Any]]:
        """The origin block as API input items, header first. POST-BREAKPOINT.

        Origin MESSAGE items carry their real metadata, so the stale-send guard keeps reading
        `metadata.ts` off them exactly as it does the canonical ones. The HEADER carries
        `metadata={}` like every other framing item: it has no message ts, and an invented one
        would be a timestamp the guard could advance a lease to.

        EVERY ITEM CARRIES `_origin`, AND DELIBERATELY NOT `_stream`. The origin block has to be
        outside `_stream` so `_evidence_hash` includes it, and inside the admission estimator's
        room-content figure so a refusal reports it as room content rather than overhead. One
        flag cannot do both: `_stream` would buy the token accounting and lose the evidence
        hash. So the estimator reads BOTH markers, and `to_input_items` removes both.

        THE STRIP IS A CONTRACT, NOT A CRASH GUARD. `_channel_input_items`
        (`openai_client/base.py:55`) rebuilds every role item from `role` + `content` alone, so
        a stray marker could never reach the API even unstripped. `to_input_items` is the seam
        that promises callers the assembler's own bookkeeping is gone; leaving a marker in its
        output would make that promise false and hand the next reader a key with no owner.
        """
        sequence = ([self.origin_header_item] if self.origin_header_item else [])
        return [{"role": item.role, "content": item.content, "metadata": dict(item.metadata),
                 "_origin": True}
                for item in (*sequence, *self.origin_items)]

    def stream_render_fields(self) -> Dict[str, Any]:
        """The stream_render telemetry payload, minus the per-turn identifiers the caller owns
        (turn_id, origin_thread_ts, trigger_ts)."""
        return {
            "channel_id": self.pinned.channel_id,
            "H": self.pinned.H,
            "periphery_floor_ts": self.periphery_floor_ts,
            "selection_version": self.selection_version,
            "inventory_start_ts": (self.pinned.coverage.start_ts
                                   if self.pinned.coverage else ""),
            "inventory_state": self.inventory_state,
            "serializer_version": self.serializer_version,
            "serializer_config_hash": self.serializer_config_hash,
            "sidecar_versions_hash": self.sidecar_versions_hash,
            "actor_map_hash": self.actor_map_hash,
            "capability_profile_hash": self.capability_profile_hash,
            "byte_count": self.byte_count,
            "origin_byte_count": self.origin_byte_count,
            "message_count": self.message_count,
            "origin_count": self.origin_count,
            "candidate_count": self.candidate_count,
            "root_count": self.root_count,
            "orphan_root_count": self.orphan_root_count,
            "stream_sha256": self.stream_sha256,
            "union_sha256": self.union_sha256,
            "receipts_included_count": len(self.receipts_included),
            "receipts_excluded_count": len(self.receipts_excluded),
            "receipts_membership_hash": self.receipts_membership_hash,
        }


# ---------------------------------------------------------------- serializer config snapshot

def _pipeline_markers() -> Tuple[str, ...]:
    try:
        from slack_client.messaging import pipeline_status_markers
    except Exception:  # noqa: BLE001
        return ()
    try:
        return tuple(sorted(str(m) for m in (pipeline_status_markers() or [])))
    except Exception:  # noqa: BLE001
        return ()


def serializer_config_snapshot() -> Dict[str, Any]:
    """The dynamic knobs the serializer's output depends on, frozen for this build.

    The grammar constants above are version-pinned, but three things the renderers consult are
    not: the provenance annotation budgets, the chrome-filter's marker list, and the ambient
    note cap. A turn whose bytes changed because someone raised a budget must be a cache MISS,
    not a silently different stream under the same hash.

    These VALUES are pinned (PinnedTuple.serializer_config), not just their hash, and everything
    derived from them — the marker lines, the chrome decisions — is derived once at pin time.
    """
    return {
        "provenance_max_entries": int(getattr(config, "tool_provenance_max_entries", 20)),
        "provenance_gist_chars": int(getattr(config, "tool_provenance_gist_chars", 80)),
        "provenance_line_budget": int(getattr(config, "tool_provenance_line_budget", 300)),
        "ambient_note_chars": AMBIENT_NOTE_CHARS,
        "image_gist_chars": IMAGE_GIST_CHARS,
        "reactions_rendered": REACTIONS_RENDERED,
        "files_marker_limit": FILES_MARKER_LIMIT,
        "chrome_markers": list(_pipeline_markers()),
        # The horizon's depth_config clause names this number, so a change to it is a change to
        # the rendered bytes and must be a cache MISS rather than a silent rewording.
        "coverage_bootstrap_days": int(getattr(config, "coverage_bootstrap_days", 90)),
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_hash(payload: Any) -> str:
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


# ---------------------------------------------------------------- Appendix A render helpers
#
# The ONLY place Appendix A bytes are produced, so a template can never drift between two
# renderers of the same thing.

# Everything except \n and \t. A control character inside a payload is either an accident or an
# attempt to break a line where the grammar says there is none; neither is worth carrying.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def escape_payload_line(line: str) -> str:
    """A7 for ONE line: strip controls, then neutralize a forged marker.

    Stripping comes first on purpose — `"\\r[END CHANNEL SUMMARY]"` starts with a control
    character, so a prefix test run before the strip would pass it through untouched and the
    line would still render as our structure.
    """
    clean = _CONTROL_RE.sub("", str(line))
    if clean.startswith(A2_RESERVED_PREFIXES):
        return ESCAPE_PREFIX + clean
    return clean


def escape_payload(text: str) -> str:
    """A7 over a whole payload, line by line. Idempotent: an escaped line no longer begins with
    a reserved prefix, so re-escaping persisted bytes changes nothing."""
    if text is None:
        return ""
    return "\n".join(escape_payload_line(line) for line in str(text).split("\n"))


def index_clause(state: Optional[str], *, depth_days: Optional[int] = None) -> str:
    """§2f's index clause for one InventoryPin state. A COMPLETE function over the six states.

    Every state renders exactly one clause and `warm` renders the empty string. `unavailable` is
    a clause now rather than a refusal: under OWNER-7 the sweep builds a thread index, so failing
    to build it degrades DISCOVERY — a recent reply under an old root may be missing — and says
    so, where it used to fail the turn closed because it defined the stream's beginning.
    """
    value = str(state or "")
    if value == INVENTORY_WARM:
        return ""
    if value in (INVENTORY_ABSENT, INVENTORY_COLD):
        return INDEX_CLAUSE_UNINDEXED
    if value == INVENTORY_UNAVAILABLE:
        return INDEX_CLAUSE_UNAVAILABLE
    if value == INVENTORY_LIMITED_DEPTH:
        days = int(depth_days if depth_days is not None
                   else getattr(config, "coverage_bootstrap_days", 90))
        return INDEX_CLAUSE_DEPTH_TEMPLATE.format(days=days)
    # THE FAIL-SAFE (§2f): limited_retention, and so is any state this function does not know.
    # Saying Slack's retention may be the limit is true whenever we cannot prove otherwise;
    # claiming our configured depth was the limit asserts a cause we do not have.
    return INDEX_CLAUSE_RETENTION


def render_reach_clause(reach_tools: Sequence[str]) -> str:
    """A1's reach segment, or the EMPTY STRING when no reach tool is exposed.

    Rendered in REACH_TOOLS order whatever order the caller supplied, so two turns with the same
    exposed set produce the same cached bytes. Joined as `a`, `a and b`, or `a, b and c`.
    """
    names = [t for t in REACH_TOOLS if t in set(reach_tools or ())]
    if not names:
        return ""
    if len(names) == 1:
        joined = names[0]
    elif len(names) == 2:
        joined = f"{names[0]} and {names[1]}"
    else:
        joined = ", ".join(names[:-1]) + f" and {names[-1]}"
    return HORIZON_REACH_TEMPLATE.format(reach_list=joined)


def render_horizon(*, floor_ts: str, inventory_state: str,
                   reach_tools: Sequence[str] = (),
                   depth_days: Optional[int] = None) -> str:
    """The A1 horizon item content. `inventory_state` is one of §2f's SIX states.

    An empty `floor_ts` selects the no-floor variant. `parse_ts` is never called on it — the
    caller guards on truthiness first, exactly as §2d requires of the sentinel it becomes.

    CARRIES NO COUNTS AND NO H. Item 0 is the head of the cached prefix, so a per-turn value here
    would invalidate the whole stream beneath it on every turn; the counts and the live edge ride
    the post-breakpoint turn-coordinates block instead, where they cost nothing.
    """
    clause = index_clause(inventory_state, depth_days=depth_days)
    reach = render_reach_clause(reach_tools)
    if not floor_ts:
        return HORIZON_TEMPLATE_NO_FLOOR.format(reach_clause=reach, index_clause=clause)
    return HORIZON_TEMPLATE.format(floor_ts=floor_ts, reach_clause=reach, index_clause=clause)


def render_origin_header(*, origin_root_ts: str, origin_count: int) -> str:
    """The A2 origin header item content."""
    return ORIGIN_HEADER_TEMPLATE.format(origin_root_ts=origin_root_ts,
                                         origin_count=int(origin_count))


def render_orphan_marker(*, root_ts: str, reach_tools: Sequence[str] = ()) -> str:
    """The A5 marker for one pre-window root. Names its tool ONLY when that tool is exposed."""
    clause = (ORPHAN_MARKER_TOOL_CLAUSE if "fetch_thread_messages" in set(reach_tools or ())
              else "")
    return ORPHAN_MARKER_TEMPLATE.format(root_ts=root_ts, tool_clause=clause)


# ---------------------------------------------------------------- serialization

def iso_minute(ts: str) -> str:
    """UTC minute stamp from the ts's SECONDS field.

    Never `int(float(ts))`: a float round at `…59.9999995` renders the wrong minute, and at a
    window boundary it renders the wrong ordering evidence to the model.
    """
    seconds, _ = parse_ts(ts)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _root_snippet(root: Optional[NormalizedMessage]) -> str:
    if root is None:
        return ""
    # Tombstone FIRST: a deleted root still has whatever text Slack left behind, and rendering
    # that as the thread's subject would describe a message that is gone.
    if root.is_tombstone:
        return DELETED_SNIPPET
    words = sanitize_name(root.text).split()
    if not words:
        return ""
    snippet = " ".join(words[:SNIPPET_WORDS])[:SNIPPET_CHARS]
    return f'~"{snippet}"' if snippet else ""


def render_header(message: NormalizedMessage, *, actor_names: Dict[str, str],
                  root: Optional[NormalizedMessage]) -> str:
    sender_id = message.sender_id or "unknown"
    name = sanitize_name(actor_names.get(message.sender_id or "", "")) or sender_id
    thread = ""
    if message.is_reply:
        thread = f" thread={message.thread_root_ts}{_root_snippet(root)}"
    flags = ""
    if message.edited_ts:
        flags += " (edited)"
    if message.is_broadcast:
        flags += " (broadcast)"
    return (f"[{iso_minute(message.ts)} {name} ({message.sender_type}) "
            f"id={sender_id} ts={message.ts}{thread}{flags}]")


def render_body(message: NormalizedMessage, actor_names: Dict[str, str]) -> List[str]:
    text = render_mentions(message.text, actor_names)
    if not text.strip():
        return []
    lines: List[str] = []
    for line in text.split("\n"):
        # A body line that starts with "[" is indistinguishable from one of OUR marker lines.
        # One backslash makes it the author's bracket again.
        lines.append(f"\\{line}" if line[:1] == "[" else line)
    return lines


def render_files_marker(message: NormalizedMessage, *,
                        limit: int = FILES_MARKER_LIMIT) -> Optional[str]:
    """`[+N files: …]`, where N is the TRUE count.

    Only `limit` of them are named, and the rest are declared as unlisted rather than left out
    of the count: a marker that said "+10 files" when a message carried twelve would be a line of
    context the model has no reason to disbelieve and no way to check. Slack caps an upload at
    ten today, so the omission clause exists for imported fixtures and for the day that changes.
    """
    total = len(message.files)
    if not total:
        return None
    named = message.files[:max(1, limit)]
    parts = [f"{sanitize_name(f.name) or 'file'} ({f.kind}) id={f.id or 'unknown'}"
             for f in named]
    omitted = total - len(named)
    if omitted > 0:
        parts.append(f"+{omitted} more not listed")
    plural = "s" if total > 1 else ""
    return f"[+{total} file{plural}: " + ", ".join(parts) + "]"


def render_reactions_marker(message: NormalizedMessage, *,
                            limit: int = REACTIONS_RENDERED) -> Optional[str]:
    if not message.reactions:
        return None
    top = sorted(message.reactions, key=lambda r: (-r.count, r.name))[:max(1, limit)]
    if not top:
        return None
    return "[reactions: " + ", ".join(
        f"{r.count}× {sanitize_name(r.name)}{' (you)' if r.mine else ''}" for r in top) + "]"


_IMAGE_URL_ID_RES = (re.compile(r"/files-pri/[^/]+-([^/?]+)/"),
                     re.compile(r"/files/[^/]+/([^/?]+)/"))


def _image_ref(row: Mapping[str, Any]) -> str:
    """A stable, short referent for an `images` row. The table has no file_id column, so the
    Slack id is parsed out of the url when it is there, and the recorded filename is the next
    best name a model can actually use."""
    url = str(row.get("url") or "")
    for pattern in _IMAGE_URL_ID_RES:
        match = pattern.search(url)
        if match:
            return sanitize_name(match.group(1))
    # A Mapping, not a dict: §1p freezes pinned rows, so a nested object arrives as a read-only
    # proxy and a `dict` test would silently lose the filename.
    metadata = row.get("metadata") or {}
    if isinstance(metadata, Mapping) and metadata.get("filename"):
        return sanitize_name(metadata["filename"])
    return sanitize_name(url.rsplit("/", 1)[-1] or url) or "image"


def _image_marker(row: Mapping[str, Any], gist_chars: int) -> Optional[Tuple[str, str]]:
    ref = _image_ref(row)
    analysis = sanitize_field(row.get("analysis") or "")
    if not analysis.strip() or is_unattended_summary(analysis):
        return None
    # Flattened before the cap: a marker is ONE line, and a newline inside it would forge a
    # second one.
    gist = " ".join(analysis.split())[:gist_chars]
    if not gist:
        return (ref, f"[image analysis ({ref}): available]")
    return (ref, f"[image analysis ({ref}): {gist}]")


def _document_marker(row: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    summary = row.get("summary")
    if not (summary or "").strip() or is_unattended_summary(summary):
        return None
    name = sanitize_name(row.get("filename") or "document") or "document"
    return (name, f"[document ({name}): summary available]")


def _ambient_marker(row: Mapping[str, Any], note_chars: int) -> Optional[Tuple[str, str]]:
    if str(row.get("status") or "") != "ready":
        return None
    if str(row.get("derivation_source") or "") == "unfurl":
        return None
    try:
        from message_processor.ambient_memory import render_artifact_note
    except Exception:  # noqa: BLE001
        return None
    note = render_artifact_note(dict(row), max_chars=note_chars)
    if not note:
        return None
    return (sanitize_name(row.get("ref") or ""), sanitize_field(note).replace("\n", " "))


def _tool_marker(tools: Sequence[Mapping[str, Any]]) -> Optional[Tuple[str, str]]:
    if not tools:
        return None
    from message_processor.tool_provenance import render_provenance_annotations
    rendered = render_provenance_annotations([dict(t) for t in tools])
    if not rendered:
        return None
    return ("", rendered)


def _render_sidecar_markers(sidecars: SidecarPin, channel_id: str,
                            cfg: Mapping[str, Any]
                            ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """The pinned sidecar rows → their marker LINES, grouped by the ts they describe.

    Called at pin time, once. The renderers this leans on (`render_provenance_annotations`,
    `render_artifact_note`) are frozen under serializer v1 and read their budgets from live
    configuration; running them here means those reads happen while the pin is being made, with
    the values recorded in `serializer_config` and hashed into `serializer_config_hash`, instead
    of on every serialization of a pin that was supposed to be settled.

    Dedup key is (channel, source_ts, kind, normalized ref) — the same file cataloged twice
    (once unattended, once by a turn that later read it) must produce one marker, not two.
    """
    grouped: Dict[str, List[Tuple[str, str, str]]] = {}
    seen: set = set()

    def add(source_ts: Optional[str], kind: str, ref: str, line: str) -> None:
        if not source_ts or not line:
            return
        key = (channel_id, str(source_ts), kind, ref)
        if key in seen:
            return
        seen.add(key)
        grouped.setdefault(str(source_ts), []).append((kind, ref, line))

    note_chars = int(cfg.get("ambient_note_chars", AMBIENT_NOTE_CHARS))
    gist_chars = int(cfg.get("image_gist_chars", IMAGE_GIST_CHARS))
    for row in sidecars.ambient_artifacts:
        marker = _ambient_marker(row, note_chars)
        if marker:
            add(row.get("source_ts"), MARKER_KIND_AMBIENT, marker[0], marker[1])
    for row in sidecars.document_extractions:
        marker = _document_marker(row)
        if marker:
            add(row.get("message_ts"), MARKER_KIND_DOCUMENT, marker[0], marker[1])
    for row in sidecars.image_analyses:
        marker = _image_marker(row, gist_chars)
        if marker:
            add(row.get("message_ts"), MARKER_KIND_IMAGE, marker[0], marker[1])
    for message_ts, tools in sidecars.tool_usage:
        marker = _tool_marker(tools)
        if marker:
            add(message_ts, MARKER_KIND_TOOL, marker[0], marker[1])

    frozen: List[Tuple[str, Tuple[str, ...]]] = []
    for source_ts, entries in grouped.items():
        lines: List[str] = []
        for _kind, _ref, line in sorted(entries, key=lambda e: (e[0], e[1])):
            lines.extend(line.split("\n"))
        frozen.append((source_ts, tuple(lines)))
    return tuple(sorted(frozen, key=lambda pair: parse_ts(pair[0])))


def _is_chrome(message: NormalizedMessage, markers: Optional[Sequence[str]] = None) -> bool:
    """Is THIS message one of our own transient UI chrome lines? Fail-open."""
    try:
        from slack_client.messaging import is_self_chrome_message
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(is_self_chrome_message(message.text, {"text": message.text},
                                           markers=markers))
    except Exception:  # noqa: BLE001
        return False


def classify_chrome(messages: Sequence[NormalizedMessage], *,
                    chrome_markers: Sequence[str]) -> FrozenSet[str]:
    """OUR OWN transient UI chrome, decided from the FROZEN marker list.

    `chrome_markers` is `serializer_config["chrome_markers"]` — the list already frozen into the
    pin. It is a REQUIRED keyword argument, never a default and never a live read: the classifier
    delegates to `is_self_chrome_message`, which consults the LIVE marker list unless one is
    supplied, so a marker-list change landing mid-turn could otherwise silently alter the bytes an
    admitted turn was already committed to.

    PURE, so the history walk can call it to evaluate its stopping predicate before any pin
    exists — chrome classification needs no receipt and no database.
    """
    return frozenset(m.ts for m in messages
                     if m.sender_type == "self" and _is_chrome(m, chrome_markers))


def eligible_for_stream(message: NormalizedMessage, pin: SidecarPin, *,
                        receipt_feature_epoch_ts: Optional[str],
                        chrome_ts: FrozenSet[str]) -> bool:
    """§2a. THE ONE PREDICATE, and it takes THE PIN — that is the whole point.

    Selection and rendering must agree by construction: a message the count admitted and the
    serializer then dropped would make the rendered window smaller than the window the floor
    claims, silently, with nothing anywhere saying so. Both callers pass the SAME pin, so they
    cannot disagree.

    It wraps the serializer's own self-role resolution rather than reimplementing it: a foreign
    message is always eligible, and one of OUR OWN is eligible exactly when `_self_role` gives it
    a role to render under.
    """
    if message.sender_type != "self":
        return True
    return _resolve_self_role(message, receipts=pin,
                              receipt_feature_epoch_ts=receipt_feature_epoch_ts,
                              chrome_ts=chrome_ts) is not None


def _self_role(message: NormalizedMessage, pinned: "PinnedTuple") -> Optional[str]:
    """The role for one of OUR OWN messages, or None when it must not be in the stream at all.

    A thin adapter over `_resolve_self_role` so the serializer keeps its landed call shape while
    `eligible_for_stream` — which runs BEFORE a PinnedTuple exists — can reach the same rule.
    """
    return _resolve_self_role(message, receipts=pinned.sidecars,
                              receipt_feature_epoch_ts=pinned.receipt_feature_epoch_ts,
                              chrome_ts=frozenset(pinned.chrome_ts))


def _resolve_self_role(message: NormalizedMessage, *, receipts: SidecarPin,
                       receipt_feature_epoch_ts: Optional[str],
                       chrome_ts: FrozenSet[str]) -> Optional[str]:
    """THE RULE, with its inputs passed explicitly rather than read off a pin.

    A FINALIZED RECEIPT is the only thing that puts one of our own messages into the stream after
    the receipts epoch. An in_flight row is a reply still being written — showing the model its
    own half-finished sentence is how a stream turns into a hall of mirrors — and chrome was never
    a reply. A post-epoch message with NO row is not evidence of a reply either: it is evidence
    that a registration was lost, and the shape heuristic that used to cover for it cannot tell a
    status card from an answer well enough to be the basis for attributing words to ourselves.
    So it is excluded, loudly.

    Only messages that predate the epoch are grandfathered, because they have no row by
    construction rather than by failure. A missing or malformed epoch cannot establish that, so it
    excludes too: the fail-closed direction is the one that omits a message we posted, not the one
    that hands the model chrome as its own past words.
    """
    receipt = receipts.receipt_for(message.ts)
    if receipt is not None:
        return ROLE_ASSISTANT if receipt.state == RECEIPT_FINALIZED else None
    if not message.text.strip() and not message.files:
        # A self message with nothing in it is a UI-helper block (Configure button, feedback
        # strip) whose blocks the normalizer does not carry. There is no reply to replay.
        return None

    epoch_ts = receipt_feature_epoch_ts
    pre_epoch = False
    reason = "no receipts epoch is recorded"
    if epoch_ts:
        try:
            pre_epoch = parse_ts(message.ts) < parse_ts(epoch_ts)
            reason = "it is newer than the receipts epoch"
        except TimestampError:
            reason = f"the recorded receipts epoch {epoch_ts!r} is unusable"
    if not pre_epoch:
        logger.warning(
            f"self message {message.channel_id}/{message.ts} has no receipt row and cannot be "
            f"grandfathered ({reason}); excluding it from the stream")
        return None
    return None if message.ts in chrome_ts else ROLE_ASSISTANT


def _render_message_item(message: NormalizedMessage, *, role: str, actor_names: Dict[str, str],
                         root: Optional[NormalizedMessage], markers: Mapping[str, Any],
                         files_limit: int, reactions_limit: int) -> StreamItem:
    """A3's message item. BYTE-FOR-BYTE THE SAME in both blocks — the only thing that differs
    between the periphery and the origin is which messages get rendered and where."""
    lines = [render_header(message, actor_names=actor_names, root=root)]
    lines.extend(render_body(message, actor_names))
    files_marker = render_files_marker(message, limit=files_limit)
    if files_marker:
        lines.append(files_marker)
    lines.extend(markers.get(message.ts, ()))
    reactions_marker = render_reactions_marker(message, limit=reactions_limit)
    if reactions_marker:
        lines.append(reactions_marker)
    return StreamItem(
        role=role,
        content="\n".join(lines),
        metadata={
            "channel_id": message.channel_id,
            "sender_id": message.sender_id,
            "ts": message.ts,
            "thread_root_ts": message.thread_root_ts,
            "sender_type": message.sender_type,
            # EDIT §2a. Slack's `edited.ts` snapshot as of THIS fetch (None when never edited) —
            # the fact an edit authorization is proved against, pinned beside the ts it is about.
            "edited_ts": message.edited_ts,
        })


def serialize_stream(pinned: PinnedTuple) -> ChannelStream:
    """The pinned tuple → the exact bytes the model sees, BOTH BLOCKS.

    A pure function of `pinned` and nothing else: no configuration read, no clock, no database,
    no Slack. Everything dynamic was resolved when the tuple was pinned, which is what makes two
    independent builds of one turn comparable and a retry a replay rather than a new question.

    TWO BLOCKS, EACH ts-ORDERED INDEPENDENTLY, AND THEY ARE NEVER RE-INTERLEAVED. The canonical
    pre-breakpoint block is the shallow periphery — byte-identical for every origin in the
    channel, which is what buys the shared cache prefix. The origin block follows the breakpoint
    and may be far older. Merging them would destroy the cache invariant and the origin's
    completeness in one move.
    """
    actor_names = pinned.actor_names
    cfg = pinned.serializer_config
    files_limit = int(cfg.get("files_marker_limit", FILES_MARKER_LIMIT))
    reactions_limit = int(cfg.get("reactions_rendered", REACTIONS_RENDERED))
    markers = dict(pinned.sidecar_markers)
    epoch = pinned.receipt_feature_epoch_ts
    chrome_ts = frozenset(pinned.chrome_ts)
    floor = pinned.periphery_floor_ts

    horizon = StreamItem(
        role=ROLE_USER,
        content=render_horizon(
            floor_ts=pinned.horizon_floor_ts,
            inventory_state=pinned.inventory_state,
            reach_tools=pinned.reach_tools,
            depth_days=cfg.get("coverage_bootstrap_days")),
        metadata={})
    end_marker = StreamItem(role=ROLE_USER, content=END_MARKER_TEXT, metadata={})

    included: List[str] = []
    excluded: List[str] = []

    def _role_for(message: NormalizedMessage) -> Optional[str]:
        """A foreign message renders `user`; one of ours renders `assistant` only with a
        finalized receipt. Membership is recorded across BOTH blocks — a receipt decides an
        origin message's role exactly as it decides a periphery one's."""
        if message.sender_type != "self":
            return ROLE_USER
        resolved = _resolve_self_role(message, receipts=pinned.sidecars,
                                      receipt_feature_epoch_ts=epoch, chrome_ts=chrome_ts)
        (excluded if resolved is None else included).append(message.ts)
        return resolved

    # ---- the canonical block: eligible periphery events at or above the floor ---------------
    # THE ONLY FILTER (§2d) is the floor. Roots and replies alike; replies ride along free. An
    # empty floor filters nothing — `parse_ts("")` must never be called.
    in_window = list(pinned.fetch_snapshot)
    if floor:
        in_window = [m for m in in_window if parse_ts(m.ts) >= parse_ts(floor)]
    in_window.sort(key=lambda m: parse_ts(m.ts))

    # TWO PASSES, because the orphan marker needs to know what WILL render before the first item
    # is built. Pass one resolves every role and records receipt membership — including the
    # exclusions, which is why eligibility is applied HERE rather than as a pre-filter: a message
    # dropped before this loop would never reach `receipts_excluded` and the ledger would stop
    # naming our own messages the stream withheld.
    roles: Dict[str, str] = {}
    for message in in_window:
        role = _role_for(message)
        if role is not None:
            roles[message.ts] = role
    periphery = [m for m in in_window if m.ts in roles]

    by_ts = {m.ts: m for m in pinned.fetch_snapshot}
    rendered_ts = set(roles)

    message_items: List[StreamItem] = []
    canonical: List[StreamItem] = [horizon]
    marked_roots: set = set()
    orphan_root_count = 0
    for message in periphery:
        role = roles[message.ts]
        root_ts = message.thread_root_ts if message.is_reply else None
        # A5: a reply whose root sits BELOW the floor is preceded, at its FIRST occurrence for
        # that root, by its own marker item — a `thread=<ts>` label pointing at a root the model
        # cannot see anywhere is precisely the shape that invites invention.
        if root_ts and root_ts not in rendered_ts and root_ts not in marked_roots:
            marked_roots.add(root_ts)
            orphan_root_count += 1
            canonical.append(StreamItem(
                role=ROLE_USER,
                content=render_orphan_marker(root_ts=root_ts, reach_tools=pinned.reach_tools),
                metadata={}))
        # THE ROOT SNIPPET ONLY WHEN THE ROOT IS RENDERED. `_root_snippet` reads neither receipts
        # nor chrome, so an out-of-window root that is our own in-flight post would otherwise leak
        # its text through this reply's header — the own-output exclusion rule defeated by a
        # header. The marker above is what identifies the thread instead.
        root = by_ts.get(root_ts or "") if (root_ts and root_ts in rendered_ts) else None
        item = _render_message_item(message, role=role, actor_names=actor_names, root=root,
                                    markers=markers, files_limit=files_limit,
                                    reactions_limit=reactions_limit)
        message_items.append(item)
        canonical.append(item)
    canonical.append(end_marker)
    items = tuple(canonical)

    # ---- the post-breakpoint origin block: the COMPLETE thread, never truncated -------------
    origin_by_ts = {m.ts: m for m in pinned.origin_snapshot}
    origin_ordered = sorted(pinned.origin_snapshot, key=lambda m: parse_ts(m.ts))
    origin_items: List[StreamItem] = []
    for message in origin_ordered:
        role = _role_for(message)
        if role is None:
            continue
        root = origin_by_ts.get(message.thread_root_ts or "") if message.is_reply else None
        origin_items.append(_render_message_item(
            message, role=role, actor_names=actor_names, root=root, markers=markers,
            files_limit=files_limit, reactions_limit=reactions_limit))

    origin_header = None
    if origin_items:
        origin_header = StreamItem(
            role=ROLE_USER,
            content=render_origin_header(origin_root_ts=str(pinned.origin_root_ts or ""),
                                         origin_count=len(origin_items)),
            metadata={})

    # ---- the two hashes, on the LANDED v1 framing ------------------------------------------
    def _feed(digest, sequence) -> int:
        total = 0
        for item in sequence:
            digest.update(f"{item.role}\n{item.content}\x00".encode("utf-8"))
            total += len(item.content.encode("utf-8"))
        return total

    stream_digest = hashlib.sha256()
    byte_count = _feed(stream_digest, items)

    # `union_sha256` CONTINUES the same running hash over the origin header and items. The header
    # is INCLUDED: it carries the root ts and the origin count, so omitting it would let two
    # origins with identical rendered messages but different headers collide on one hash.
    union_digest = hashlib.sha256()
    _feed(union_digest, items)
    origin_sequence = ([origin_header] if origin_header else []) + origin_items
    origin_byte_count = _feed(union_digest, origin_sequence)

    # CANONICALIZED TO UNIQUE TIMESTAMPS BEFORE ANYTHING READS THEM. `_role_for` records a ts on
    # every RENDER, and RULING-1 deliberately renders an origin message that also sits in the
    # periphery TWICE — so the raw lists double-count exactly the messages the duplication is
    # about. A receipt is a fact about a MESSAGE, not about how many times the message appeared,
    # so the counts and the hash are per-ts: §8's "counted by UNIQUE ts". Doing it here, once,
    # keeps the three consumers — the two counts and the membership hash — from disagreeing.
    included_ts = sorted(set(included), key=parse_ts)
    excluded_ts = sorted(set(excluded), key=parse_ts)
    membership = _sha256(
        "included:" + ",".join(included_ts) + ";excluded:" + ",".join(excluded_ts))
    return ChannelStream(
        pinned=pinned,
        items=items,
        horizon_item=horizon,
        end_marker_item=end_marker,
        message_items=tuple(message_items),
        origin_header_item=origin_header,
        origin_items=tuple(origin_items),
        stream_sha256=stream_digest.hexdigest(),
        union_sha256=union_digest.hexdigest(),
        byte_count=byte_count,
        origin_byte_count=origin_byte_count,
        message_count=len(message_items),
        origin_count=len(origin_items),
        candidate_count=len(pinned.fetch_snapshot),
        root_count=len({m.ts for m in periphery if not m.is_reply}),
        orphan_root_count=orphan_root_count,
        periphery_floor_ts=floor,
        selection_version=pinned.selection_version,
        inventory_state=pinned.inventory_state,
        serializer_version=pinned.serializer_version,
        serializer_config_hash=pinned.serializer_config_hash,
        sidecar_versions_hash=pinned.sidecar_versions_hash,
        actor_map_hash=pinned.actor_map_hash,
        capability_profile_hash=pinned.capability_profile_hash,
        receipts_included=tuple(included_ts),
        receipts_excluded=tuple(excluded_ts),
        receipts_membership_hash=membership,
    )


# ---------------------------------------------------------------- fetch + discovery

def _web(client: Any, name: str) -> Optional[Callable[..., Any]]:
    app = getattr(client, "app", None)
    web = getattr(app, "client", None) if app is not None else None
    method = getattr(web, name, None)
    if callable(method):
        return method
    method = getattr(client, name, None)
    return method if callable(method) else None


def _normalize_page(client: Any, raw: Iterable[Dict[str, Any]], *, channel_id: str,
                    team_id: str, origin: str, floor_ts: str, floor_inclusive: bool,
                    high: str) -> List[NormalizedMessage]:
    """Normalize a page and apply the LOCAL window predicate.

    Slack's own boundary semantics are not trusted here: we fetch inclusive=true and decide
    membership ourselves with the numeric comparator, so a message exactly on the floor is kept
    at an inclusive floor and dropped at an exclusive one, which is what the flag means.

    A payload the normalizer DECLINES (a join notice, a reply-count notification) is not a
    message and is skipped. A payload it cannot place in time is a different thing entirely: the
    window predicate has no answer for it, so the turn fails closed rather than render a stream
    with a hole in it that nobody can see.
    """
    out: List[NormalizedMessage] = []
    for payload in raw:
        try:
            message = normalize_slack_message(client, payload, channel_id=channel_id,
                                             origin=origin, team_id=team_id)
        except TimestampError as e:
            raise StreamTimestampError(
                f"{channel_id} {origin} page carries an unusable timestamp: {e}") from e
        if message is None:
            continue
        if not in_window(message.ts, floor_ts, floor_inclusive, high):
            continue
        out.append(message)
    return out


async def _fetch_replies(client: Any, *, channel_id: str, team_id: str, root_ts: str,
                         floor_ts: str, floor_inclusive: bool, high: str,
                         budget: FetchBudget) -> List[NormalizedMessage]:
    method = _web(client, "conversations_replies")
    if method is None:
        raise HistoryFetchError("no conversations.replies method on this client")
    raw = await page_messages(method, channel_id=channel_id, oldest=floor_ts, latest=high,
                              inclusive=True, budget=budget,
                              extra_params={"ts": root_ts}, label="channel replies")
    return _normalize_page(client, raw, channel_id=channel_id, team_id=team_id,
                           origin=ORIGIN_REPLIES, floor_ts=floor_ts,
                           floor_inclusive=floor_inclusive, high=high)


def _dedup(messages: Iterable[NormalizedMessage]) -> List[NormalizedMessage]:
    """One record per ts, replies-origin winning, whole record — never a field merge.

    A broadcast is delivered twice (as a top-level history entry and as a reply); the replies
    copy is the one that knows its thread, and it carries the broadcast flag itself, so keeping
    the whole record is both simpler and more accurate than stitching two together.
    """
    chosen: Dict[str, NormalizedMessage] = {}
    for message in messages:
        current = chosen.get(message.ts)
        if current is None or (current.origin != ORIGIN_REPLIES
                               and message.origin == ORIGIN_REPLIES):
            chosen[message.ts] = message
    return sorted(chosen.values(), key=lambda m: parse_ts(m.ts))


async def _build_actor_map(client: Any, messages: Sequence[NormalizedMessage], *,
                           skip_ids: FrozenSet[str] = frozenset(),
                           max_remote_lookups: Optional[int] = None,
                           stats: Optional[Dict[str, Any]] = None,
                           ) -> Tuple[Tuple[str, str], ...]:
    """Names for every actor these messages mention or attribute.

    Read-only by contract: resolving a name for a message we are merely READING must not create
    a user row or bump anyone's last_seen.

    `skip_ids` are ids ANOTHER phase already ATTEMPTED — the ids it TRIED, not merely the ones it
    succeeded at. An id the periphery attempted and failed renders raw in both blocks; retrying
    it here would make one block's rendering of an actor depend on which origin was built, and
    the periphery's bytes must not move with the origin.

    `stats` is filled in place by the resolver with the budget it was given and the remote
    lookups it spent, so the caller can hand the REMAINDER to the next phase.
    """
    names: Dict[str, str] = {}
    human_ids: List[str] = []
    self_name = sanitize_name(
        getattr(client, "bot_handle", None)
        or (config.bot_name_aliases or ["assistant"])[0])
    for message in messages:
        sender = message.sender_id
        if not sender:
            continue
        # SKIPPED FIRST, WHATEVER THE SENDER TYPE. `skip_ids` used to guard only the human
        # branch — the one that spends remote lookups — which left a real hole: a bot appearing
        # in the periphery WITHOUT a raw name but in the origin WITH one had its name added by
        # the origin phase, and `render_header` then rendered that name into the PERIPHERY. Two
        # origins would produce different pre-breakpoint bytes, which is the one thing the
        # stable prefix may never do. The budget was never the only reason to skip: a periphery
        # actor's rendering must not depend on which thread the turn originated in.
        if sender in skip_ids:
            continue
        if message.sender_type == "self":
            names[sender] = self_name
        elif message.sender_type == "other_bot":
            if message.raw_bot_name:
                names[sender] = message.raw_bot_name
        elif sender not in names and sender not in human_ids:
            human_ids.append(sender)
    for message in messages:
        for mention in message.mention_ids:
            if (mention not in names and mention not in human_ids
                    and mention not in skip_ids):
                human_ids.append(mention)
    if human_ids:
        resolver = getattr(client, "resolve_usernames", None)
        api_client = getattr(getattr(client, "app", None), "client", None)
        if callable(resolver):
            budget = (ACTOR_REMOTE_LOOKUP_DEFAULT if max_remote_lookups is None
                      else max(0, int(max_remote_lookups)))
            # CAPABILITY IS DETECTED, NOT INFERRED FROM AN EXCEPTION. Catching `TypeError` to
            # mean "older signature" also catches a TypeError raised INSIDE a real resolver —
            # and the retry then ran the work twice and handed the origin phase the full default
            # budget instead of its remainder, silently overspending the cap. Reading the
            # signature asks the question directly, and every bound the resolver DOES support is
            # still passed.
            kwargs: Dict[str, Any] = {}
            try:
                accepted = inspect.signature(resolver).parameters
                supports_kwargs = any(pm.kind == inspect.Parameter.VAR_KEYWORD
                                      for pm in accepted.values())
                if supports_kwargs or "max_remote_lookups" in accepted:
                    kwargs["max_remote_lookups"] = budget
                if supports_kwargs or "stats" in accepted:
                    kwargs["stats"] = stats
            except (TypeError, ValueError):
                # An unintrospectable callable: pass the bounds and let it speak for itself.
                kwargs = {"max_remote_lookups": budget, "stats": stats}
            try:
                resolved = await resolver(human_ids, api_client, **kwargs)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"actor map name resolution failed: {e}")
                resolved = {}
            for uid, name in (resolved or {}).items():
                clean = sanitize_name(name)
                if clean:
                    names[uid] = clean
    return tuple(sorted(names.items()))


async def _drop_dead_root(db: Any, *, team_id: str, channel_id: str, root_ts: str,
                          event_ts: Optional[str]) -> None:
    """Compare-and-delete the activity row of a root Slack says is gone (W2-LIVE-2).

    HYGIENE, NOT CORRECTNESS — the build has already succeeded without this root, so a failed
    cleanup warns and the turn stands; the next turn retries it. What the delete buys is an END
    to the failure: the row is exempt from the floor, so an undeleted dead root re-fires the same
    refused fetch on every turn forever.
    """
    remover = getattr(db, "delete_thread_activity_if_unchanged_async", None)
    if not callable(remover):
        return
    try:
        await remover(team_id, channel_id, root_ts, if_event_ts_equals=event_ts)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"activity row for {channel_id}/{root_ts} not cleaned up: {e}")


async def _clear_dirty(db: Any, *, team_id: str, channel_id: str, root_ts: str,
                       event_ts: Optional[str]) -> None:
    clearer = getattr(db, "clear_thread_dirty_async", None)
    if not callable(clearer):
        return
    try:
        await clearer(team_id, channel_id, root_ts, if_event_ts_equals=event_ts)
    except Exception as e:  # noqa: BLE001
        # A dirty flag that fails to clear costs one redundant fetch next turn. It must never
        # cost the turn that just succeeded.
        logger.warning(f"dirty flag for {channel_id}/{root_ts} not cleared: {e}")


async def _gather_or_cancel(tasks: List[asyncio.Task]) -> List[Any]:
    """gather, but a failure cancels and AWAITS the siblings before propagating — an orphaned
    replies fetch would keep spending the turn's page budget after the turn had already failed."""
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


# ---------------------------------------------------------------- the build

def _emit_stream_render(result: StreamBuildResult, *, turn_id: Optional[str],
                        origin_root_ts: Optional[str], trigger_ts: Optional[str]) -> None:
    """One line per BUILD, for the TURN population only. TAKES THE WHOLE CARRIER.

    `result.stream.stream_render_fields()` supplies everything derivable from the build;
    `reselected`, `anchor_advanced` and all THREE page counts come off the carrier, because none
    of them is knowable at serialization — the two booleans postdate the bytes and the page
    counts postdate the pin the bytes were made from.

    A build with NO turn_id is not a turn: the dev probes and out-of-process rebuilds all come
    through here, and their rows would join to nothing and inflate the build count. So they emit
    nothing, and the production path — which always passes one — cannot lose its row silently.
    """
    if not turn_id:
        return
    from message_processor import participation_telemetry

    try:
        participation_telemetry.stream_render(
            turn_id=turn_id, origin_thread_ts=origin_root_ts, trigger_ts=trigger_ts,
            reselected=result.reselected, anchor_advanced=result.anchor_advanced,
            history_pages=result.pages.history, reply_pages=result.pages.reply,
            origin_pages=result.pages.origin,
            **result.stream.stream_render_fields())
    except Exception as e:  # noqa: BLE001
        logger.debug(f"stream_render telemetry not emitted: {e}")


def guaranteed_eligible_root(m: NormalizedMessage, chrome_ts: FrozenSet[str]) -> bool:
    """A root that NO pin lookup can later remove from the window.

    The walk cannot count ELIGIBLE roots — eligibility needs the pin, which does not exist yet —
    and must not count RAW roots, because a run of our own receiptless posts could satisfy a raw
    count while leaving fewer than TARGET survivors, with older eligible roots never fetched.

    THE PROOF, in one sentence: receipt state can only ever ADD one of OUR OWN roots to the
    eligible set, never subtract a foreign one — so more than CEILING guaranteed-eligible roots
    is a LOWER BOUND on the eligible root count, and re-anchoring to TARGET is therefore certain
    and under-fill impossible. Both predicates are available pre-pin: `sender_type` comes from the
    normalizer's bot-id test, and chrome is classified from the frozen marker list.
    """
    return (not m.is_reply
            and m.sender_type != "self"
            and m.ts not in chrome_ts)


async def prepare_channel_turn(*, client: Any, db: Any, team_id: str, channel_id: str, h: str,
                               frontier: int = 0, drain_timeout: Optional[float] = None,
                               barrier_context: Optional[Dict[str, Any]] = None,
                               skip_dev_barrier: bool = False,
                               ) -> PreparedTurn:
    """§4.4 steps 1-4, and NOTHING may fetch until it returns.

    Its existence is structural rather than stylistic. The origin fetch runs CONCURRENTLY with
    the periphery build, so if the watermark drain lived inside that build, the origin's first
    `conversations.replies` could begin before the drain had resolved — the one ordering P1
    established. Splitting the pre-fetch steps into their own awaited phase makes that violation
    impossible rather than merely discouraged.

    `skip_dev_barrier=True` skips `dev_barriers.post_admission()` entirely — the barrier writes
    files and can block, and omitting `barrier_context` does NOT skip it. The pure
    reconsideration snapshot passes True (STALE_RECONSIDERATION §4c); every production turn
    keeps the default.
    """
    checked_h = _checked_ts(h, "H")

    # The index must have caught up to everything inside the window. `drain` is the whole gate:
    # it raises on a failed-unrepaired write at or below the frontier, and ignores anything above
    # it, whose event sits outside this window by construction.
    await admission_watermark.drain(channel_id, frontier, timeout=drain_timeout)

    # FREEZE THE SERIALIZER CONFIG FIRST. It is pure configuration, so it is freezable at turn
    # start — and freezing it here is what makes the walk's stop predicate legal, because that
    # predicate needs the frozen chrome-marker list.
    serializer_config = serializer_config_snapshot()

    # READ 1 STAGE 1 — the anchor and the inventory, BOTH PINNED, and nothing else. Activity and
    # receipt roots need a window, and the window is not known until the walk has run.
    anchor_payload = await db.read_channel_window_anchor_async(team_id, channel_id)
    anchor = (anchor_payload or {}).get("anchor") or None
    inventory_row = (anchor_payload or {}).get("inventory") or None
    coverage = None
    if inventory_row:
        coverage = InventoryPin(
            start_ts=_checked_ts(inventory_row.get("inventory_start_ts"), "inventory_start_ts"),
            status=str(inventory_row.get("bootstrap_status") or ""),
            reason=inventory_row.get("reason"))

    floor_read = None
    if anchor and int(anchor.get("selection_version", -1)) == SELECTION_VERSION:
        raw_floor = str(anchor.get("floor_ts") or "") or None
        # CHECKED HERE, where the row is read. An unparseable stored floor otherwise reaches a
        # raw `parse_ts` deep in selection and raises the normalizer's `TimestampError`, which
        # is not a `ChannelStreamError` — so the turn takes the generic handler and tells the
        # user "something went wrong" instead of the honest `stream_data_invalid` notice, and
        # the ledger records no code at all. A malformed floor is a malformed record of ours.
        floor_read = _checked_ts(raw_floor, "persisted window anchor floor") if raw_floor else None

    # CAPTURED BEFORE THE FETCH, and the ordering is the whole point: `reconcile_window` returns
    # False without touching anything when the channel's generation has MOVED, so capturing it
    # afterwards would compare the post-fetch value against itself, the guard could never fire,
    # and a live event arriving mid-fetch would be silently clobbered by staler data.
    generation = actor_tail_module.generation(channel_id)

    if not skip_dev_barrier:
        await dev_barriers.post_admission(**{**(barrier_context or {}), "channel_id": channel_id,
                                             "H": checked_h, "floor_ts": floor_read or ""})
    return PreparedTurn(team_id=team_id, channel_id=channel_id, h=checked_h, frontier=frontier,
                        floor_read=floor_read, coverage=coverage, generation=generation,
                        selection_version=SELECTION_VERSION,
                        serializer_config=serializer_config)


async def fetch_origin_thread(client: Any, channel_id: str, origin_root_ts: Optional[str],
                              h: str, budget: FetchBudget,
                              trigger_ts: Optional[str]) -> OriginFetch:
    """§4.4 step 5a. The origin thread, PAGED TO COMPLETION and never truncated.

    `trigger_ts` is what lets this evaluate §2e's empty-origin taxonomy: an empty origin is legal
    in exactly ONE shape — a top-level trigger whose own message Slack has not yet propagated to
    `conversations.replies`. A reply-triggered turn whose established thread comes back empty
    FAILS, because silently replacing a real thread with one message is the corruption OWNER-2
    forbids.
    """
    if not origin_root_ts:
        return OriginFetch(origin_root_ts=None, messages=(), pages=0, empty_fallback=False,
                           deadline_at=budget.deadline_at)

    method = _web(client, "conversations_replies")
    if method is None:
        raise OriginFetchError("no conversations.replies method on this client")

    top_level = str(origin_root_ts) == str(trigger_ts or "")
    try:
        raw = await page_messages(method, channel_id=channel_id, latest=h, inclusive=True,
                                  budget=budget, label="origin thread",
                                  extra_params={"ts": str(origin_root_ts)})
    except Exception as e:  # noqa: BLE001 — re-raised as the NAMED failure below
        code = _origin_error_code(e)
        # THE ALLOWLIST: only `thread_not_found` on a top-level trigger reaches the fallback.
        # Anything unrecognised — and `None`, which is what a malformed page or a timeout gives —
        # fails closed, which is the allowlist's default.
        if code == "thread_not_found" and top_level:
            # THE PAGES THE BUDGET ACTUALLY CHARGED, not zero. Slack was called and the attempt
            # was paid for; reporting 0 would make a fallback turn look free in telemetry and
            # hide a channel that takes this path on every turn.
            return OriginFetch(origin_root_ts=str(origin_root_ts), messages=(),
                               pages=budget.pages_used,
                               empty_fallback=True, deadline_at=budget.deadline_at)
        raise OriginFetchError(
            f"origin thread {channel_id}/{origin_root_ts} could not be read completely: {e}",
            code=code) from e

    messages = _dedup(_normalize_page(client, raw, channel_id=channel_id,
                                      team_id=getattr(client, "self_team_id", "") or "",
                                      origin=ORIGIN_REPLIES, floor_ts="0",
                                      floor_inclusive=True, high=h))
    if not messages:
        if top_level:
            return OriginFetch(origin_root_ts=str(origin_root_ts), messages=(),
                               pages=budget.pages_used,
                               empty_fallback=True, deadline_at=budget.deadline_at)
        raise OriginFetchError(
            f"origin thread {channel_id}/{origin_root_ts} came back empty for a reply-triggered "
            "turn; a real thread is never replaced by a single message", code=None)
    return OriginFetch(origin_root_ts=str(origin_root_ts), messages=tuple(messages),
                       pages=budget.pages_used, empty_fallback=False,
                       deadline_at=budget.deadline_at)


def _origin_error_code(error: BaseException) -> Optional[str]:
    """§2e's PINNED extraction, in this order and NO OTHER. Wrapping otherwise destroys the very
    string the taxonomy branches on.

    THE TYPES ARE CHECKED, not merely the attribute. Reading `.code` off ANY exception makes the
    code a duck-typed property of the whole error space: an unrelated failure that happens to
    carry `code="thread_not_found"` — a wrapper, a library error, one of our own future
    exceptions — would be read as Slack's answer and silently drop a root, or reach the origin
    fallback. Both branches decide whether a turn refuses or proceeds, so the only shapes that
    may speak here are the two that genuinely carry a SLACK code:

      1. `HistoryPageError` — the pager already parsed Slack's error into `.code`;
      2. a raw `SlackApiError` — extracted through `slack_error_code`.

    Anything else returns None, which is not a taxonomy match and therefore fails closed.
    """
    if isinstance(error, HistoryPageError):
        code = getattr(error, "code", None)
        return str(code) if code else None
    if isinstance(error, SlackApiError):
        return slack_error_code(error) or None
    return None


async def build_channel_pin(prepared: PreparedTurn, *, client: Any, db: Any,
                            reach_tools: Tuple[str, ...] = (),
                            capability_profile_hash: str = "", tool_schema_version: str = "",
                            probe: bool = False,
                            deadline_at: float,
                            history_budget: Optional[FetchBudget] = None,
                            reply_budget: Optional[FetchBudget] = None,
                            ) -> SharedChannelPin:
    """§4.4 steps 3a, 5b, 6-8, READ 2a, 10 and 11a — everything ORIGIN-INDEPENDENT.

    `probe=True` skips the dirty compare-and-clear, the one write left in this phase.
    """
    team_id, channel_id, h = prepared.team_id, prepared.channel_id, prepared.h
    cfg = prepared.serializer_config
    markers = cfg.get("chrome_markers", ())
    target = int(config.channel_window_target)
    ceiling = int(config.channel_window_ceiling)
    floor_read = prepared.floor_read

    # THE DIRECT SEAM IS VALIDATED TOO. `build_channel_pin` is callable on its own — the probe
    # does exactly that — so an injected budget must agree with the deadline this phase was
    # handed, or the component it bounds is running on a clock nobody else shares.
    #
    # A MISSING DEADLINE IS REFUSED OUTRIGHT. `deadline_at=None` would let this phase build its
    # own `total_seconds` budgets, each starting its own window — the three-independent-clocks
    # defect, reintroduced through the seam that exists to test it is gone. The parameter is
    # required in the signature; this makes it required in fact.
    if deadline_at is None:
        raise ChannelStreamError(
            "build_channel_pin requires an absolute deadline_at; without one each component "
            "starts its own window and a turn can spend its budget three times over")
    for name, injected in (("history_budget", history_budget), ("reply_budget", reply_budget)):
        if injected is not None and injected.deadline_at != deadline_at:
            raise ChannelStreamError(
                f"{name} carries deadline_at={injected.deadline_at!r} but this phase was given "
                f"{deadline_at!r}; every component of one turn shares ONE absolute clock")
    history_budget = history_budget or FetchBudget(deadline_at=deadline_at)
    reply_budget = reply_budget or FetchBudget(deadline_at=deadline_at, page_ceiling=None)

    candidates: List[NormalizedMessage] = []
    chrome_ts: set = set()
    reached_floor = True

    # STEP 3a — F NEWER THAN H. A floor above H selects nothing, so there is no window to fetch:
    # the history walk, stage-2 discovery and the reply fan-out are ALL skipped. Only the history
    # call being skipped is not enough — a dirty root from the index would fan out into
    # `conversations.replies` with the same inverted bounds.
    inverted = bool(floor_read) and parse_ts(floor_read) > parse_ts(h)

    if not inverted:
        # STEP 5b — THE HISTORY WALK, page by page, stopping at whichever bound comes FIRST.
        method = _web(client, "conversations_history")
        if method is None:
            raise HistoryFetchError("no conversations.history method on this client")
        seen_roots: set = set()
        reached_floor = floor_read is None
        async for page in iter_pages(method, channel_id=channel_id,
                                     oldest=floor_read, latest=h, inclusive=True,
                                     budget=history_budget, label="channel history"):
            fresh = _normalize_page(client, page, channel_id=channel_id, team_id=team_id,
                                    origin=ORIGIN_HISTORY,
                                    floor_ts=floor_read or "0", floor_inclusive=True, high=h)
            new = [m for m in fresh if m.ts not in {c.ts for c in candidates}]
            candidates.extend(new)
            # Classify chrome on the NEW messages only, memoized by ts — ONE OPERATIONAL
            # classification per message, which is what lets the stop predicate be evaluated
            # without a pin.
            chrome_ts |= classify_chrome(new, chrome_markers=markers)
            for m in new:
                if guaranteed_eligible_root(m, frozenset(chrome_ts)):
                    seen_roots.add(m.ts)
            if len(seen_roots) > ceiling:
                # EARLY STOP APPLIES WITH A STORED FLOOR TOO. A stale floor far below a busy
                # channel's recent traffic would otherwise page all the way down to it before
                # re-anchoring — the deep walk this design exists to refuse to pay for.
                reached_floor = False
                break
            if floor_read is not None:
                reached_floor = True

    candidates = _dedup(candidates)

    discovery: Dict[str, Any] = {"activity_roots": {}, "receipt_roots": ()}
    if not inverted:
        # STEP 7 — DISCOVERY, windowed on the EFFECTIVE floor: `F` only when the walk actually
        # reached it, otherwise the walk's early-stop floor. Binding it to a stale stored floor
        # is precisely the fan-out the staged discovery exists to prevent, reached through the
        # index instead of through Slack.
        #
        # IT ALWAYS RUNS. `conversations.history` returns top-level messages only, so a channel
        # whose sole in-window activity is a reply under an older root walks back ZERO events —
        # and that is exactly the channel the index exists to serve.
        #
        # THE THREE CASES KEY ON THE WALK'S OUTCOME, in this order:
        #   1. the walk REACHED F                  -> F. A walk that paged [F, H] to exhaustion
        #      and found nothing reached it, so a complete EMPTY walk lands here too.
        #   2. otherwise, it fetched anything      -> the oldest fetched EVENT, root or reply.
        #      This is the early-stop case, and F is deliberately NOT used even when set: a
        #      stale stored floor is exactly the fan-out staged discovery exists to prevent.
        #   3. otherwise                           -> the inventory floor, or "0" with no
        #      inventory row. A zero-event walk has no oldest event to floor on, and F is
        #      necessarily unset here (an early stop needs candidates). Bounded and cheap
        #      precisely where it applies: such a channel has almost nothing to return.
        if reached_floor and floor_read:
            effective_floor = floor_read
        elif candidates:
            effective_floor = min((m.ts for m in candidates), key=parse_ts)
        else:
            effective_floor = prepared.coverage.start_ts if prepared.coverage else "0"
        discovery = await db.read_channel_discovery_roots_async(
            team_id, channel_id, floor_ts=effective_floor, high_ts=h)

        # STEP 7a / STEP 8 — the reply fan-out, entered only when there is a candidate root.
        # THE SHORTCUT IS CHECKED HERE, AFTER DISCOVERY, and it needs BOTH conditions: zero
        # history events AND zero discovery candidates. Keying it on the empty walk alone
        # skipped the read above and lost the very reply the index had recorded.
        roots = _discovery_roots(candidates, discovery, h)
        if roots:
            semaphore = asyncio.Semaphore(max(1, int(config.reply_fetch_concurrency)))

            # SEEN vs UNSEEN is decided from the pages ALREADY FETCHED, before the fan-out
            # starts: a root is SEEN when a normalized candidate carries its ts, or when a
            # fetched reply or broadcast names it as their thread root. Anything else is known
            # only to the DB.
            seen_root_ids = ({m.ts for m in candidates}
                             | {m.thread_root_ts for m in candidates if m.thread_root_ts})

            async def _one(root_ts: str) -> Tuple[str, Optional[List[NormalizedMessage]]]:
                async with semaphore:
                    try:
                        return root_ts, await _fetch_replies(
                            client, channel_id=channel_id, team_id=team_id, root_ts=root_ts,
                            floor_ts=effective_floor, floor_inclusive=True, high=h,
                            budget=reply_budget)
                    except Exception as e:  # noqa: BLE001 — re-raised unless it is THE case
                        # THREAD_NOT_FOUND ON AN UNSEEN ROOT: DROP, DON'T DIE. The code comes
                        # from the same wrap order §2e's taxonomy uses, never from a string
                        # match. An unseen root is old — a fresh one would sit in history — so
                        # this is no propagation race, and nothing visible vouches for it: there
                        # is no partial periphery to present dishonestly, which is what failure
                        # condition 4 exists to prevent. A SEEN root answering the same code is
                        # live evidence contradicting the API and still fails the turn, because
                        # dropping it would hide a broadcast the room can see.
                        if (_origin_error_code(e) == "thread_not_found"
                                and root_ts not in seen_root_ids):
                            return root_ts, None
                        raise

            tasks = [asyncio.ensure_future(_one(root)) for root in roots]
            indexed = dict(discovery.get("activity_roots") or {})
            for root_ts, messages in await _gather_or_cancel(tasks):
                if messages is None:
                    logger.warning(
                        f"channel {channel_id}: thread root {root_ts} is gone from Slack "
                        "(thread_not_found, and nothing in this window references it) — "
                        "dropping it from this build")
                    if not probe and root_ts in indexed:
                        await _drop_dead_root(db, team_id=team_id, channel_id=channel_id,
                                              root_ts=root_ts, event_ts=indexed[root_ts])
                    continue
                candidates.extend(messages)
                chrome_ts |= classify_chrome(messages, chrome_markers=markers)
                # COMPARE-AND-CLEAR. Without it every dirty root stays dirty forever — a dirty
                # row is exempt from the floor, so an uncleared historical root is re-fetched on
                # every turn for the life of the channel. The compare is what keeps it correct:
                # a root whose stored ts MOVED saw a mutation this fetch did not.
                if not probe and root_ts in indexed:
                    await _clear_dirty(db, team_id=team_id, channel_id=channel_id,
                                       root_ts=root_ts, event_ts=indexed[root_ts])
            candidates = _dedup(candidates)

    # READ 2a — THE RENDER PIN over the PERIPHERY candidate identities.
    raw = await db.read_channel_sidecars_for_async(
        team_id, channel_id, sorted({m.ts for m in candidates}))
    sidecars = _freeze_sidecars(raw)

    # STEP 10 — SELECT, computed FROM the pin and never before it.
    epoch = sidecars.receipt_feature_epoch_ts
    frozen_chrome = frozenset(chrome_ts)
    eligible = [m for m in candidates
                if eligible_for_stream(m, sidecars, receipt_feature_epoch_ts=epoch,
                                       chrome_ts=frozen_chrome)]
    floor_selected = _select_floor(eligible, floor_read=floor_read, target=target,
                                   ceiling=ceiling, inverted=inverted)
    periphery = ([m for m in eligible if parse_ts(m.ts) >= parse_ts(floor_selected)]
                 if floor_selected else ([] if inverted else eligible))
    periphery.sort(key=lambda m: parse_ts(m.ts))

    # STEP 11a — THE PERIPHERY ACTOR MAP, resolved ONCE and frozen. The resolver MUTATES an
    # in-memory cache, so two origin pins resolving separately could render two different
    # periphery names for one id — pre-breakpoint bytes moving with the origin, which is exactly
    # what the stable prefix forbids.
    stats: Dict[str, Any] = {"budget": ACTOR_REMOTE_LOOKUP_DEFAULT,
                             "remote_lookups": 0, "attempted_ids": set()}
    actor_map = await _build_actor_map(client, periphery, stats=stats)
    remaining = max(0, int(stats["budget"]) - int(stats["remote_lookups"]))

    return SharedChannelPin(
        team_id=team_id, channel_id=channel_id, h=h, deadline_at=deadline_at,
        serializer_config=cfg, generation=prepared.generation, floor_read=floor_read,
        periphery_floor_ts=floor_selected,
        reselected=(floor_selected or None) != (floor_read or None),
        periphery_candidates=tuple(candidates), periphery=tuple(periphery),
        periphery_sidecars=sidecars, periphery_chrome_ts=frozen_chrome,
        actor_map=actor_map,
        actor_ids_attempted=frozenset(stats.get("attempted_ids") or ()),
        actor_lookups_remaining=remaining, coverage=prepared.coverage,
        selection_version=prepared.selection_version, reach_tools=tuple(reach_tools),
        capability_profile_hash=capability_profile_hash,
        tool_schema_version=tool_schema_version,
        pages=SharedPageCounts(history=history_budget.pages_used,
                               reply=reply_budget.pages_used))


def _select_floor(eligible: Sequence[NormalizedMessage], *, floor_read: Optional[str],
                  target: int, ceiling: int, inverted: bool) -> str:
    """§2d's count rule. THE COUNT RUNS OVER ROOTS; THE FLOOR SELECTS EVENTS.

    Returns `F'`, or the empty-floor sentinel `""`. A stored floor is NEVER reset: a channel
    whose events were all deleted keeps its floor and renders nothing above it.
    """
    if inverted:
        return floor_read or ""
    if not eligible:
        # Floor-never-backward governs: only a channel that has never had a floor AND has no
        # events reaches the sentinel.
        return floor_read or ""
    roots = sorted((m for m in eligible if not m.is_reply), key=lambda m: parse_ts(m.ts))
    if len(roots) > ceiling:
        # The ts of the OLDEST root among the newest TARGET. Strictly greater — a window sitting
        # exactly AT the ceiling keeps its floor and its cached prefix.
        return roots[-target].ts
    if floor_read:
        return floor_read
    # A cold channel starts at its oldest eligible EVENT — event, not root, so a channel whose
    # oldest reachable message is a reply starts there.
    return min((m.ts for m in eligible), key=parse_ts)


def _discovery_roots(candidates: Sequence[NormalizedMessage], discovery: Mapping[str, Any],
                     high: str) -> List[str]:
    """Every thread that might hold a message in this window, from all four sources.

    KNOWN ROOTS ARE RETAINED, DELIBERATELY. A root already present among the candidates is still
    fanned out: `conversations.history` returns top-level messages only, so a root we have does
    not mean we have its REPLIES, and those replies are exactly what rides free inside the window
    (§4.4 step 8 collects from all four sources with no known-filter). Filtering here would drop
    the fan-out for every root the walk happened to return, which is most of them.

    This used to end in `[r for r in roots if r not in known or True]` — an always-true filter
    that computed a `known` set and then ignored it, leaving a reader unable to tell whether the
    retention was intended or a bug. It is intended; the set is gone and the rule is written down.
    """
    roots: List[str] = []
    seen: set = set()
    high_key = parse_ts(high)

    def add(candidate: Optional[str], what: str) -> None:
        if not candidate:
            return
        value = str(candidate)
        if value in seen:
            return
        if parse_ts(_checked_ts(value, what)) > high_key:
            return
        seen.add(value)
        roots.append(value)

    for message in candidates:
        if message.reply_count or message.latest_reply:
            add(message.root_ts, "page parent root_ts")
        if message.is_reply:
            add(message.thread_root_ts, "page reply thread_root_ts")
    for root in (discovery.get("activity_roots") or {}):
        add(root, "activity root_ts")
    for root in (discovery.get("receipt_roots") or ()):
        add(root, "receipt thread_root_ts")
    return roots


async def build_origin_pin(shared: SharedChannelPin, origin_fetch: OriginFetch, *,
                           db: Any, client: Any = None) -> Tuple[PinnedTuple, int]:
    """The per-origin pin AND that origin's page count. It performs NO Slack fetch.

    IT VALIDATES THE DEADLINE IT CAN ACTUALLY SEE. This is a WIRING assertion, not a timeout: it
    catches a caller that built the two components against two clocks, which is the defect the
    one-deadline rule exists to prevent and which no later symptom would name.
    """
    # A `None` deadline is a MISMATCH, not an exemption. It means the origin was fetched on a
    # budget with its own `total_seconds` window rather than the shared clock — precisely the
    # shape this assertion exists to catch — so skipping the check when it is None would waive
    # the rule for the only caller that could break it.
    #
    # AND EQUAL-BUT-ABSENT IS NOT AGREEMENT. `None == None` passes an equality test while
    # describing two components that each started their own clock, so the missing case is
    # refused before the comparison rather than sliding through it.
    if shared.deadline_at is None or origin_fetch.deadline_at is None:
        raise ChannelStreamError(
            "both the shared periphery and the origin fetch must carry an absolute deadline "
            f"(shared={shared.deadline_at!r}, origin={origin_fetch.deadline_at!r}); two absent "
            "deadlines are two independent clocks, not one shared one")
    if origin_fetch.deadline_at != shared.deadline_at:
        raise ChannelStreamError(
            "the origin fetch and the shared periphery were built against DIFFERENT deadlines "
            f"({origin_fetch.deadline_at!r} vs {shared.deadline_at!r}); the two components must "
            "share one absolute clock or a turn can spend its budget twice")

    origin_messages = tuple(origin_fetch.messages)
    periphery_ids = {m.ts for m in shared.periphery_candidates}
    origin_only = sorted({m.ts for m in origin_messages} - periphery_ids)

    # READ 2b — the ORIGIN-ONLY ids. The two id sets are disjoint by construction.
    if origin_only:
        origin_raw = await db.read_channel_sidecars_for_async(
            shared.team_id, shared.channel_id, origin_only)
        merged = merge_sidecar_pins(shared.periphery_sidecars, _freeze_sidecars(origin_raw),
                                    ids=sorted(periphery_ids | set(origin_only)))
    else:
        merged = shared.periphery_sidecars

    # The chrome memo EXTENDS over the origin candidates with the SAME frozen markers. An
    # origin-only message is subject to the same eligibility rule, so classifying only the
    # periphery would let an old pre-epoch chrome message of ours render inside the origin block.
    markers = shared.serializer_config.get("chrome_markers", ())
    chrome_ts = shared.periphery_chrome_ts | classify_chrome(
        [m for m in origin_messages if m.ts not in periphery_ids], chrome_markers=markers)

    # STEP 11b — ORIGIN-ONLY actors, on the REMAINING budget, never re-resolving a periphery id.
    actor_map = dict(shared.actor_map)
    if client is not None and origin_messages:
        # THE RULE: the origin may not name anyone the periphery's own map was built from.
        #
        # That covers more than the ids the periphery ATTEMPTED or RESOLVED, and the gap is where
        # the defect lived: a bot seen in the room without a `raw_bot_name` is never attempted
        # (bots never reach `human_ids`) and never resolved (nothing lands in `names`), so it fell
        # straight through to the origin phase, which had a message carrying its name and added
        # it — into bytes the periphery had already rendered. Mentions are included because they
        # render into periphery bodies.
        #
        # THE SUBJECT IS THE SELECTION, NOT THE CANDIDATES. Only rendered messages produce
        # pre-breakpoint bytes, so the selection is the exact set the invariant needs; candidates
        # is a superset that would also gag the origin about actors the room never shows, losing
        # post-breakpoint naming for nothing. `shared.periphery` is FIXED at compose time — step
        # 10 selects it, step 11a builds the actor map from it, and `SharedChannelPin` is frozen
        # before any origin phase runs — so the rule cannot wobble if the fetch order changes.
        periphery_actor_ids = frozenset(
            {m.sender_id for m in shared.periphery if m.sender_id}
            | {uid for m in shared.periphery for uid in m.mention_ids})
        origin_names = await _build_actor_map(
            client, origin_messages,
            skip_ids=(shared.actor_ids_attempted | frozenset(actor_map)
                      | periphery_actor_ids),
            max_remote_lookups=shared.actor_lookups_remaining)
        for uid, name in origin_names:
            actor_map.setdefault(uid, name)

    pinned = PinnedTuple(
        team_id=shared.team_id, channel_id=shared.channel_id,
        window=(shared.periphery_floor_ts, True), H=shared.h,
        fetch_snapshot=tuple(shared.periphery_candidates),
        origin_root_ts=origin_fetch.origin_root_ts,
        origin_snapshot=origin_messages,
        periphery_floor_ts=shared.periphery_floor_ts,
        selection_version=shared.selection_version,
        reach_tools=shared.reach_tools,
        sidecar_versions_hash=merged.versions_hash,
        actor_map=tuple(sorted(actor_map.items())),
        actor_map_hash=_stable_hash(sorted(actor_map.items())),
        serializer_version=SERIALIZER_VERSION,
        serializer_config_hash=_stable_hash(dict(shared.serializer_config)),
        capability_profile_hash=shared.capability_profile_hash,
        tool_schema_version=shared.tool_schema_version,
        coverage=shared.coverage,
        receipt_feature_epoch_ts=merged.receipt_feature_epoch_ts,
        receipt_map=tuple((r.ts, r.state, r.turn_id, r.thread_root_ts) for r in merged.receipts),
        sidecars=merged,
        serializer_config=dict(shared.serializer_config),
        chrome_ts=chrome_ts)
    return pinned, origin_fetch.pages


async def build_channel_stream(*, client: Any, db: Any, team_id: str, channel_id: str,
                               h: str, frontier: int = 0,
                               origin_root_ts: Optional[str] = None,
                               capability_profile_hash: str = "",
                               tool_schema_version: str = "",
                               drain_timeout: Optional[float] = None,
                               reach_tools: Tuple[str, ...] = (),
                               history_budget: Optional[FetchBudget] = None,
                               reply_budget: Optional[FetchBudget] = None,
                               origin_budget: Optional[FetchBudget] = None,
                               barrier_context: Optional[Dict[str, Any]] = None,
                               turn_id: Optional[str] = None,
                               trigger_ts: Optional[str] = None) -> StreamBuildResult:
    """Build one channel turn's stream. THE COMPOSITION of the phases above.

    Its signature, its position in the turn and its observable behaviour are what every caller
    sees; internally it awaits `prepare_channel_turn`, then runs the origin fetch CONCURRENTLY
    with `build_channel_pin`, then `build_origin_pin`. That split is what lets one periphery be
    serialized under two origins.

    The three budget parameters are TEST-ONLY SEAMS. Production passes none: the builder takes
    ONE absolute deadline the instant the prepare phase returns and constructs all three from it.
    """
    prepared = await prepare_channel_turn(
        client=client, db=db, team_id=team_id, channel_id=channel_id, h=h, frontier=frontier,
        drain_timeout=drain_timeout, barrier_context=barrier_context)

    # TAKEN HERE, NOT BEFORE THE DRAIN — "deadline at fetch start" means at FETCH start, and a
    # drain that waits must not eat the fetch budget it is not part of.
    # EVERY SUPPLIED BUDGET MUST CARRY THE SAME NON-None ABSOLUTE DEADLINE. The legacy
    # `total_seconds` form is NOT accepted here, and that is the whole point: three budgets built
    # with `total_seconds=60` each record their own start instant, so a turn could spend 180
    # seconds while every budget reported itself within bounds. Accepting an all-None set would
    # readmit exactly the defect the shared deadline exists to remove — the seam would look
    # validated while the components ran on three independent clocks.
    supplied = [b for b in (history_budget, reply_budget, origin_budget) if b is not None]
    deadlines = {b.deadline_at for b in supplied}
    if supplied and (None in deadlines or len(deadlines) > 1):
        raise ChannelStreamError(
            "injected fetch budgets must each carry the SAME absolute deadline_at; the "
            "total_seconds form gives every component its own clock and is not accepted here "
            f"(got {sorted(str(d) for d in deadlines)})")
    deadline_at = next((d for d in deadlines if d is not None),
                       time.monotonic() + float(config.fetch_retry_total_seconds))
    origin_budget = origin_budget or FetchBudget(deadline_at=deadline_at, page_ceiling=None)

    # The two components run CONCURRENTLY, and whichever fails first CANCELS the other and
    # AWAITS it before propagating — an orphaned fetch would keep spending wall clock after the
    # turn had already failed.
    shared_task = asyncio.ensure_future(build_channel_pin(
        prepared, client=client, db=db, reach_tools=reach_tools,
        capability_profile_hash=capability_profile_hash,
        tool_schema_version=tool_schema_version, deadline_at=deadline_at,
        history_budget=history_budget, reply_budget=reply_budget))
    origin_task = asyncio.ensure_future(fetch_origin_thread(
        client, channel_id, origin_root_ts, h, origin_budget, trigger_ts))
    shared, origin_fetch = await _gather_or_cancel([shared_task, origin_task])

    pinned, origin_pages = await build_origin_pin(shared, origin_fetch, db=db, client=client)
    stream = serialize_stream(pinned)

    # PERSIST F' — skipped when the floor is the sentinel or unchanged. A failure is a WARNING,
    # never a turn failure: the bytes are already correct and the next turn re-derives the floor.
    anchor_advanced = False
    if shared.periphery_floor_ts and shared.periphery_floor_ts != (shared.floor_read or ""):
        try:
            anchor_advanced = bool(await db.advance_channel_window_anchor_async(
                team_id, channel_id, shared.periphery_floor_ts, shared.selection_version))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"window anchor for {channel_id} not advanced: {e}")

    # The actor tail is fed the PERIPHERY only, self messages filtered — feeding it the whole
    # origin would let an old origin message evict the recent other-bot record the gate's
    # continuation veto depends on.
    actor_tail_module.reconcile_window(
        channel_id,
        [actor_tail_module.tail_record(m) for m in shared.periphery
         if m.sender_type != "self"],
        window=(shared.periphery_floor_ts, True, h),
        expected_generation=shared.generation)

    result = StreamBuildResult(
        stream=stream, reselected=shared.reselected, anchor_advanced=anchor_advanced,
        pages=PageCounts(history=shared.pages.history, reply=shared.pages.reply,
                         origin=origin_pages))
    _emit_stream_render(result, turn_id=turn_id, origin_root_ts=origin_root_ts,
                        trigger_ts=trigger_ts)
    return result


async def build_reconsideration_snapshot(*, client: Any, db: Any, team_id: str, channel_id: str,
                                         trigger_ts: Optional[str] = None,
                                         origin_root_ts: Optional[str] = None,
                                         capability_profile_hash: str = "",
                                         tool_schema_version: str = "",
                                         reach_tools: Tuple[str, ...] = (),
                                         drain_timeout: Optional[float] = None,
                                         ) -> StreamBuildResult:
    """The PURE snapshot seam for stale reconsideration (STALE_RECONSIDERATION §4c).

    Composed exactly like `build_channel_stream`: one shared absolute deadline taken when the
    prepare phase returns, periphery and origin built CONCURRENTLY from the same phase methods.
    H and the frontier come from a FRESH atomic `admission_watermark.pin()`, so the rebuilt
    window contains everything admitted since the original turn pinned its own — including the
    message that suppressed the draft.

    Permitted side effects, exactly: in-memory username/display-name cache fills (the probe
    acknowledges the same, `tools/stream_probe.py`). Forbidden, and structurally absent here:
    the durable anchor write, dirty-state clearing and dead-root drops (`probe=True`),
    actor-tail writes, telemetry emission of any kind, and the dev barrier
    (`skip_dev_barrier=True` — the barrier writes files and can block).

    A deadline miss or any fetch failure propagates; the caller treats it as a context-rebuild
    failure (§4f), never as a reason to post unexamined.
    """
    pin = admission_watermark.pin(channel_id, trigger_ts)
    prepared = await prepare_channel_turn(
        client=client, db=db, team_id=team_id, channel_id=channel_id, h=pin.h,
        frontier=pin.frontier, drain_timeout=drain_timeout, skip_dev_barrier=True)

    # ONE absolute deadline, taken after the drain exactly as the full builder takes it.
    deadline_at = time.monotonic() + float(config.fetch_retry_total_seconds)
    origin_budget = FetchBudget(deadline_at=deadline_at, page_ceiling=None)

    shared_task = asyncio.ensure_future(build_channel_pin(
        prepared, client=client, db=db, reach_tools=reach_tools,
        capability_profile_hash=capability_profile_hash,
        tool_schema_version=tool_schema_version, probe=True, deadline_at=deadline_at))
    origin_task = asyncio.ensure_future(fetch_origin_thread(
        client, channel_id, origin_root_ts, pin.h, origin_budget, trigger_ts))
    shared, origin_fetch = await _gather_or_cancel([shared_task, origin_task])

    pinned, origin_pages = await build_origin_pin(shared, origin_fetch, db=db, client=client)
    stream = serialize_stream(pinned)

    # No anchor persist, no actor-tail reconcile, no `_emit_stream_render` — the snapshot reads
    # the world and writes nothing durable about having done so.
    return StreamBuildResult(
        stream=stream, reselected=shared.reselected, anchor_advanced=False,
        pages=PageCounts(history=shared.pages.history, reply=shared.pages.reply,
                         origin=origin_pages))


def _freeze_sidecars(payload: Optional[Dict[str, Any]]) -> SidecarPin:
    """One sidecar read → the frozen pin, timestamps checked as they go in.

    A row with no usable ts is NOT skipped here. A receipt or an activity row we quietly dropped
    would decide the role of one of our own messages, or lose a whole thread from discovery, with
    nothing anywhere saying so; `SidecarPin` raises instead and the turn refuses honestly.
    """
    data = payload or {}
    coverage_row = data.get("coverage") or None
    coverage = None
    if coverage_row and coverage_row.get("inventory_start_ts"):
        coverage = InventoryPin(
            start_ts=_checked_ts(coverage_row["inventory_start_ts"], "inventory_start_ts"),
            status=str(coverage_row.get("bootstrap_status") or ""),
            reason=coverage_row.get("reason"))
    receipts = tuple(
        ReceiptRec(ts=_checked_ts(row.get("message_ts"), "receipt message_ts"),
                   state=str(row.get("state") or ""),
                   turn_id=row.get("turn_id"), thread_root_ts=row.get("thread_root_ts"),
                   receipt_class=row.get("receipt_class"))
        for row in (data.get("receipts") or []))
    activity = tuple(data.get("activity") or [])
    activity_roots = tuple(_checked_ts(row.get("root_ts"), "activity root_ts")
                           for row in activity)
    raw_window = data.get("window") or (coverage.start_ts if coverage else "0", True)
    return SidecarPin(
        window=(str(raw_window[0]), bool(raw_window[1])),
        receipts=receipts,
        receipt_feature_epoch_ts=data.get("receipt_feature_epoch_ts"),
        coverage=coverage,
        activity_roots=activity_roots,
        activity_event_ts=tuple(zip(activity_roots,
                                    (row.get("last_index_event_ts") for row in activity))),
        image_analyses=tuple(data.get("image_analyses") or []),
        document_extractions=tuple(data.get("document_extractions") or []),
        ambient_artifacts=tuple(data.get("ambient_artifacts") or []),
        tool_usage=tuple((str(ts), tuple(tools))
                         for ts, tools in (data.get("tool_usage") or {}).items()),
        versions_hash=str(data.get("versions_hash") or ""),
    )
