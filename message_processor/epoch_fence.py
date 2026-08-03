"""Dev-only epoch fence — SLIM build (Docs/specs/EPOCH_FENCE_SPEC.md, owner decision 2026-07-31).

The live battery reads a channel a previous battery wrote to. The fence gives one dev channel a
pinned starting state and a per-case boundary, so two runs are comparable. **Test infrastructure —
it must cost production nothing.**

WHAT SLIM IS. Two mechanisms, and only two:

  * **OVERLAYS.** While a fence is active, the POLICY / SELECTION / PREFERENCE state for the fenced
    channel — settings, standing policy, channel memory, the window anchor, response feedback —
    is served from and written to an in-memory store keyed by the current sub-epoch. No production
    row is read or written for those tables. Observable facts about real Slack objects (receipts,
    the activity index, artifacts, coverage) still persist, because those messages really exist.
    **RULED, and it is the sharp edge of that rule:** a fenced battery turn meeting a dead root
    DELETES the production `channel_thread_activity` row through the compare-and-delete accessor,
    and that is CORRECT — the row records a fact about the world, the fact changed, and the delete
    is recoverable through the index. It gets no overlay redirect, for the same reason the
    inventory gets none.
  * **EFFECT REFUSAL.** `authorize_effect` refuses a channel-visible write when the fenced scope is
    not open for business — armed, closing, expired — or when the calling task carries a STALE
    epoch identity.

WHAT SLIM IS NOT, deliberately. There is no channel admission gate ahead of turn dispatch, no
channel identity in the post gate or the ingress tracker, no per-channel task ownership index, and
no channel-scoped drain. **A turn already in flight when a case boundary lands can still post into
the channel**, and the battery is expected to tolerate that. The full design in the spec closes
that hole by threading channel identity through several production structures; the owner declined
to spend that on a harness.

THE NO-OP CONTRACT. Without `DEV_EPOCH_FENCE_ENABLE` nothing here does anything: no control file is
opened, no fence table is created, no watcher task exists, and every entry point returns on a single
empty-dict test. `_FENCES` can only become non-empty by way of the watcher, which only exists under
the flag — so `if not _FENCES` IS the proof, not a flag read standing in for one.

THE ALLOWLIST IS TENANT-SCOPED AND BUILT AT BOOT. A bare channel id is not a scope: the same id
could in principle name a channel in another workspace. `init_epoch_scopes(self_team_id)` runs once
`auth.test` has landed; before that any fence request REFUSES, and a request naming a channel
outside the allowlist is rejected LOUDLY — raise and ERROR log, never silently ignored.
"""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from config import DEV_EPOCH_FENCE_ENV, dev_epoch_fence_requested
from logger import setup_logger
from slack_client.normalizer import parse_ts

logger = setup_logger(name="slack_bot.EpochFence")

# --- runtime gating (§7.1a) ---------------------------------------------------------------

#: Re-exported so callers that already hold this module need not also reach into `config`.
#: The flag's MEANING is defined once, in `config.dev_epoch_fence_requested` — it has to live
#: outside this module, because it is what decides whether this module is imported at all.
ENV_ENABLE = DEV_EPOCH_FENCE_ENV
ENV_CONTROL_DIR = "DEV_EPOCH_FENCE_DIR"

#: The only channels a fence may ever name. Hardcoded, not configurable: a fence that could be
#: pointed at an arbitrary channel by an env var is one typo away from silencing production.
FENCEABLE_CHANNELS: Tuple[str, ...] = ("C0BKX77NU66",)

#: The three fence constants slim uses. (The spec's fourth, the scoped-drain deadline, has no
#: meaning here — slim has no drain.) Both are owner-approved values transcribed from the P4
#: alignment; the watcher's poll cadence is read from `dev_barriers._POLL_SECONDS` at each use.
HEARTBEAT_SECONDS = 30.0
EXPIRY_SECONDS = 300.0

#: The steering text a fenced channel starts every case with.
EPOCH_STEERING_FIXTURE = "No standing guidance for this channel."

#: Overlay memory rows are numbered from HERE, far above any production `channel_memory.id`.
#: Two accessors — `update_channel_memory_async` and `delete_channel_memory_async` — take a row id
#: and no channel, so they cannot be routed by scope. An id band routes them unambiguously instead:
#: an id at or above this floor belongs to an overlay, and an id below it can only be production.
OVERLAY_MEMORY_ID_BASE = 900_000_000

#: The sub-epoch id an ARMED fence carries. A fence between `activate` and its first `advance` has
#: no case open, but overlay reads still need a key — this one is reset by the first advance, so
#: nothing written against it can leak into case 1.
ARMED_EPOCH = "<armed>"


class EpochFenceError(Exception):
    """Base for every fence refusal."""


class EpochScopeError(EpochFenceError):
    """A fence was requested for a scope that is not allowlisted, or before boot initialisation."""


class EpochEffectRefused(EpochFenceError):
    """A channel-visible effect was attempted against a fenced scope that will not have it."""


def fence_enabled() -> bool:
    """True when `DEV_EPOCH_FENCE_ENABLE` is set to something truthy.

    Delegates to `config.dev_epoch_fence_requested`, which the import gates in `main.py`,
    `slack_client/messaging.py` and `database.py` also call. One definition, so this module's
    idea of "enabled" can never disagree with the predicate that decided to import it.
    """
    return dev_epoch_fence_requested()


_SCOPES: Optional[frozenset] = None


def init_epoch_scopes(self_team_id: Optional[str]) -> frozenset:
    """Build the tenant-scoped allowlist, ONCE, after `auth.test` has resolved the team.

    Cannot be a module constant: the team id does not exist until boot. Idempotent, so a second
    boot path calling it is harmless — but a call with a DIFFERENT team is a bug worth hearing
    about, so it raises.
    """
    global _SCOPES
    team = (self_team_id or "").strip()
    if not team:
        raise EpochScopeError(
            "init_epoch_scopes needs the workspace team id from auth.test; got nothing. "
            "A bare channel id is not a scope.")
    scopes = frozenset((team, ch) for ch in FENCEABLE_CHANNELS)
    if _SCOPES is not None and _SCOPES != scopes:
        raise EpochScopeError(
            f"epoch fence scopes already initialised for a different workspace: "
            f"{sorted(_SCOPES)} != {sorted(scopes)}")
    _SCOPES = scopes
    logger.info(f"Epoch fence scopes initialised: {sorted(scopes)}")
    return scopes


def epoch_scopes() -> frozenset:
    """The allowlist. Raises when a fence call arrives before boot initialisation."""
    if _SCOPES is None:
        raise EpochScopeError(
            "epoch fence used before init_epoch_scopes(); no workspace is allowlisted yet")
    return _SCOPES


def require_scope(team_id: Optional[str], channel_id: Optional[str]) -> Tuple[str, str]:
    """Validate a requested fence scope, LOUDLY. Returns the normalised `(team_id, channel_id)`.

    An out-of-scope request is never "no fence": it is an operator pointing the harness at a
    channel it must never touch, and the only honest answer is to say so and refuse.
    """
    scope = ((team_id or "").strip(), (channel_id or "").strip())
    if scope not in epoch_scopes():
        logger.error(
            f"Epoch fence REFUSED for out-of-scope {scope}; allowlist is {sorted(epoch_scopes())}")
        raise EpochScopeError(f"epoch fence scope not allowlisted: {scope}")
    return scope


# --- state (§7.1c-bis) --------------------------------------------------------------------


class FenceKey(NamedTuple):
    """`lease_id` is IN the key because case ids repeat: every battery names its cases `case-01`,
    `case-02`. Keying on the case id alone would let run N+1 read run N's leftover overlay rows."""
    team_id: str
    channel_id: str
    lease_id: str
    test_epoch_id: str


@dataclass(frozen=True)
class MemoryRow:
    id: int
    content: str
    scope: str            # 'channel' | 'workspace' | 'policy' — the landed vocabulary
    author: Optional[str]
    created_ts: str
    updated_ts: str


@dataclass(frozen=True)
class FeedbackRow:
    channel_id: str
    thread_ts: Optional[str]
    message_ts: str
    user_id: str
    signal: int
    source: str
    created_ts: str


@dataclass(frozen=True)
class EpochFixture:
    """The pinned starting state. Two runs are comparable only if this is byte-identical, so the
    payload is enumerated rather than described."""
    memory: Tuple[MemoryRow, ...]
    steering: str
    channel_settings: Mapping[str, Any]
    human_bot_ids: Tuple[str, ...]
    fixture_version: str
    classification_overlay_version: str


@dataclass(frozen=True)
class EpochContext:
    """What a stamped task carries. `test_epoch_id`/`start_ts` are None while ARMED — there is no
    case open yet, and `authorize_effect` refuses everything in that window."""
    lease_id: str
    test_epoch_id: Optional[str]
    start_ts: Optional[str]
    fixture_version: str
    classification_overlay_version: str


@dataclass(frozen=True)
class ActiveFence:
    team_id: str
    channel_id: str
    context: EpochContext
    fixture: EpochFixture
    overlay: "EpochOverlayStore"
    #: 'armed' | 'active' | 'closing' | 'invalidated'. Only 'active' authorizes work.
    #: 'invalidated' is the DENY-ONLY state a restart installs: the lease row says a battery was
    #: running and the overlays that battery built did not survive the process, so the scope must
    #: stay shut. It is not merely a durable flag — without a runtime fence carrying it, `_FENCES`
    #: would be empty and every effect would authorize, which is the opposite of what the boot
    #: log promises.
    state: str
    expiry_ts: str

    @property
    def key(self) -> FenceKey:
        return FenceKey(self.team_id, self.channel_id, self.context.lease_id,
                        self.context.test_epoch_id or ARMED_EPOCH)


#: THE process-wide registry. Empty unless the flag is set AND the watcher installed a fence, which
#: is what makes every entry point below a single dict test in production.
_FENCES: Dict[Tuple[str, str], ActiveFence] = {}


def install_fence(fence: ActiveFence) -> None:
    _FENCES[(fence.team_id, fence.channel_id)] = fence
    logger.info(
        f"Epoch fence INSTALLED for {fence.channel_id} lease={fence.context.lease_id} "
        f"state={fence.state} case={fence.context.test_epoch_id}")


def remove_fence(team_id: str, channel_id: str) -> None:
    fence = _FENCES.pop((team_id, channel_id), None)
    if fence is not None:
        fence.overlay.drop()
        logger.info(f"Epoch fence REMOVED for {channel_id} lease={fence.context.lease_id}")


def set_fence_state(team_id: str, channel_id: str, state: str) -> None:
    fence = _FENCES.get((team_id, channel_id))
    if fence is not None:
        _FENCES[(team_id, channel_id)] = replace(fence, state=state)


def set_fence_expiry(team_id: str, channel_id: str, expiry_ts: str) -> None:
    """Refresh the runtime expiry from the DURABLE lease row.

    The harness heartbeats the row, not this object, so a fence whose expiry was frozen at install
    time would be torn down 300 seconds into a battery the harness was faithfully keeping alive.
    The row is the truth about the lease's life; this carries that truth into the registry the
    effect check actually reads.
    """
    fence = _FENCES.get((team_id, channel_id))
    if fence is not None and fence.expiry_ts != expiry_ts:
        _FENCES[(team_id, channel_id)] = replace(fence, expiry_ts=expiry_ts)


def install_denied_fence(team_id: str, channel_id: str, lease_id: str, expiry_ts: str) -> None:
    """Install a DENY-ONLY fence for a scope a restart invalidated.

    Everything about it exists to refuse: `state='invalidated'` fails `authorize_effect`'s state
    check, and the case columns are None so nothing can match on identity either. The overlay is
    the plain fixture, which keeps a shut channel's reads off production policy rows as well.
    Cleared only by an explicit `release` — an invalidated battery is a thing a human must look at.
    """
    fixture = build_fixture()
    install_fence(ActiveFence(
        team_id=team_id, channel_id=channel_id,
        context=EpochContext(lease_id=lease_id, test_epoch_id=None, start_ts=None,
                             fixture_version=fixture.fixture_version,
                             classification_overlay_version=fixture.classification_overlay_version),
        fixture=fixture, overlay=EpochOverlayStore(fixture),
        state="invalidated", expiry_ts=expiry_ts))


def advance_fence(team_id: str, channel_id: str, test_epoch_id: str, start_ts: str,
                  expiry_ts: Optional[str] = None) -> ActiveFence:
    """Open a new case. The OUTGOING sub-epoch's overlay entries are reclaimed here — case N's
    remembered facts must not be visible to case N+1, which is the whole point of sub-epochs."""
    fence = _FENCES[(team_id, channel_id)]
    fence.overlay.reset_subepoch(fence.key)
    moved = replace(
        fence,
        context=replace(fence.context, test_epoch_id=test_epoch_id, start_ts=start_ts),
        state="active",
        expiry_ts=expiry_ts or fence.expiry_ts)
    _FENCES[(team_id, channel_id)] = moved
    logger.info(f"Epoch fence ADVANCED for {channel_id} case={test_epoch_id} start_ts={start_ts}")
    return moved


def active_fence(team_id: Optional[str], channel_id: Optional[str]) -> Optional[ActiveFence]:
    """The fence covering `(team, channel)`, or None.

    A `None` team resolves through the channel: slim does not thread `ToolContext.team_id` (that
    was part of the production surface the owner cut), and the registry can only ever hold scopes
    from the single allowlisted workspace. A team that is present and DIFFERENT is not fenced —
    the fence is one workspace's fence.
    """
    if not _FENCES or not channel_id:
        return None
    fence = None
    for (team, channel), candidate in _FENCES.items():
        if channel == channel_id and (not team_id or team == team_id):
            fence = candidate
            break
    return fence


def _expired(fence: ActiveFence, now: Optional[str] = None) -> bool:
    try:
        return parse_ts(now or f"{time.time():.6f}") >= parse_ts(fence.expiry_ts)
    except Exception:  # noqa: BLE001 — an unparseable expiry is a dead fence
        return True


# --- context propagation (§7.1g) -----------------------------------------------------------

_EPOCH_CTX: ContextVar[Optional[EpochContext]] = ContextVar("epoch_ctx", default=None)


@contextmanager
def epoch_scope(ctx: Optional[EpochContext]):
    """SET the epoch contextvar for this task and reset it on exit.

    Reset via the token, never by setting None — a nested scope must restore its parent rather
    than erase it. `asyncio.create_task` copies the current context, so anything a stamped turn
    spawns inherits the stamp without a second site.
    """
    token = _EPOCH_CTX.set(ctx)
    try:
        yield ctx
    finally:
        _EPOCH_CTX.reset(token)


def current_epoch() -> Optional[EpochContext]:
    return _EPOCH_CTX.get()


def stamp_for(team_id: Optional[str], channel_id: Optional[str]) -> Optional[EpochContext]:
    """The context a dispatch site should enter, or None when the channel is unfenced."""
    fence = active_fence(team_id, channel_id)
    return fence.context if fence is not None else None


def stamp_current_task(team_id: Optional[str], channel_id: Optional[str]
                       ) -> Optional[EpochContext]:
    """Stamp the RUNNING task with the fence covering `(team, channel)`, and never unset it.

    Set-without-reset is correct precisely here and nowhere else: a channel turn is dispatched as
    its own task, so the context it mutates is that task's private copy and dies with it. That is
    what lets ONE stamping site cover the turn AND everything it spawns — `asyncio.create_task`
    copies the current context, so a detached image job or a background build inherits the epoch
    without a second site to forget.

    Returns None on an unfenced channel, having touched nothing.
    """
    ctx = stamp_for(team_id, channel_id)
    if ctx is not None:
        _EPOCH_CTX.set(ctx)
    return ctx


# --- authorization (§7.1g) ------------------------------------------------------------------


def authorize_effect(team_id: Optional[str], channel_id: Optional[str], *, site: str) -> None:
    """Raise `EpochEffectRefused` when a channel-visible write must not happen.

    TWO CHECKS, IN THIS ORDER.

    1. **STATE.** The fence must be exactly `active` with a case open, and unexpired. `armed` (the
       battery has claimed the channel but opened no case), `closing` and an expired lease all
       refuse. This check is what makes expiry safe: a case boundary does not rotate the lease id,
       so a task from the dying epoch still MATCHES on identity and only the state turns it away.
    2. **IDENTITY.** A caller that CARRIES an epoch context must match the live fence's
       `(lease_id, test_epoch_id)` — never the case id alone, because case ids repeat and a task
       stranded from an expired lease whose case was also `case-01` would post run N's work into
       run N+1.

    **SLIM ACCEPTS AN UNSTAMPED CALLER.** The full design fails closed on a missing contextvar,
    which is only safe once every dispatch site stamps; slim stamps the channel turn and lets that
    cover what the turn spawns, so refusing everything unstamped would refuse the battery's own
    Bolt callbacks. The consequence is stated plainly in this module's docstring: a pre-fence
    in-flight turn can still post into the channel.
    """
    if not _FENCES:
        return
    fence = active_fence(team_id, channel_id)
    if fence is None:
        return
    ctx = fence.context
    if fence.state != "active" or ctx.test_epoch_id is None or ctx.start_ts is None:
        raise EpochEffectRefused(
            f"{site}: epoch fence on {fence.channel_id} is {fence.state!r} with no case open; "
            f"no channel work is authorized until the next advance")
    if _expired(fence):
        raise EpochEffectRefused(
            f"{site}: epoch fence on {fence.channel_id} (lease {ctx.lease_id}) has EXPIRED at "
            f"{fence.expiry_ts}; the harness must re-acquire before the channel takes writes")
    live = current_epoch()
    if live is not None and (live.lease_id, live.test_epoch_id) != (ctx.lease_id,
                                                                   ctx.test_epoch_id):
        raise EpochEffectRefused(
            f"{site}: task carries epoch ({live.lease_id}, {live.test_epoch_id}) but "
            f"{fence.channel_id} is fenced at ({ctx.lease_id}, {ctx.test_epoch_id})")


# --- the window predicate (§7.1e) -----------------------------------------------------------


def in_epoch(message: Any, start_ts: Optional[str]) -> bool:
    """BOTH clauses: the message is at/after `start_ts` AND its ROOT is too.

    The root clause is the load-bearing half, not belt-and-braces. A ts-only check admits a REPLY
    posted after the boundary under a root created before it — the thread is pre-epoch and its
    history is out of the case — and admits that reply's `thread_broadcast`, which arrives on the
    history page as though it were top-level. Both are exactly the contamination a case boundary
    exists to prevent.

    `start_ts=None` (unfenced, or armed) returns True: the predicate is a no-op off-fence.

    NOT YET WIRED IN THE SLIM BUILD — the stream's filter points live in
    `message_processor/channel_stream.py`, which this wave does not own. See the handoff note in
    the build report.
    """
    if start_ts is None:
        return True
    floor = parse_ts(start_ts)
    root = getattr(message, "root_ts", None) or getattr(message, "ts", None)
    return parse_ts(getattr(message, "ts")) >= floor and parse_ts(root) >= floor


# --- the fixture (§7.1d) ---------------------------------------------------------------------


def _fixture_channel_settings(now_ts: str) -> Dict[str, Any]:
    """Every semantic column of `channel_settings` AT HEAD, at its global default.

    `NULL` means INHERIT, and that is the honest baseline: a fenced channel behaves like a channel
    nobody has configured, which is the comparable starting state a battery needs.

    `enable_web_search`, `enable_mcp` and `image_model` became real columns in the shallow-stream
    respec's W4, and they are NULL here like every other one: a fenced channel is a channel nobody
    has configured. They still do not decide the fixture hash — the capability values enter it
    from the globals a fenced turn really inherits (see `_resolved_profile`), not from the spec's
    aspiration that the fence turns web search and MCP off. Turning them off is a `.env` decision
    for the dev bot running the battery, and the hash will report whichever way that went.
    """
    return {
        "response_mode": None,
        "reply_in_channel": None,
        "participation_level": None,
        "snoozed_until": None,
        "muted_threads": [],
        "model": None,
        "reasoning_effort": None,
        "verbosity": None,
        "ambient_memory": None,
        "enable_web_search": None,
        "enable_mcp": None,
        "image_model": None,
        "updated_ts": now_ts,
        "updated_by": "epoch_fence",
    }


def _resolved_profile(config: Any) -> Dict[str, Any]:
    """The seven `CHANNEL_CAPABILITY_KEYS`, RESOLVED against the globals, in key order.

    Resolved rather than raw: two runs whose stored settings are identically NULL but whose global
    defaults differ behave differently, and a hash over the NULLs would call them comparable.

    EVERY VALUE HERE MIRRORS `BotConfig._channel_capability_profile` (`config.py:1122`), which is
    what a channel turn actually runs on. An earlier draft pinned web search and MCP to False
    because §7.1d says the fence turns them off — but slim overlays no capability columns
    (`channel_settings` has none at HEAD), so a fenced turn inherits the globals like any other.
    A fixture hash that claimed those were off would have said two runs were comparable while the
    turns underneath them reached the network differently, which is the one thing this hash exists
    to rule out. It reports what the run DID, not what the full design would have arranged.
    """
    return {
        "model": getattr(config, "gpt_model", None),
        "reasoning_effort": getattr(config, "default_reasoning_effort", None),
        "verbosity": getattr(config, "default_verbosity", None),
        "enable_web_search": getattr(config, "enable_web_search", None),
        "enable_mcp": getattr(config, "mcp_enabled_default", None),
        "image_model": getattr(config, "image_model", None),
        "enable_code_interpreter": getattr(config, "enable_code_interpreter", None),
    }


def _resolved_settings(config: Any, baseline: Mapping[str, Any]) -> Dict[str, Any]:
    """The settings baseline with every inherit-NULL replaced by the global it inherits.

    Row METADATA is excluded — `updated_ts`, `updated_by` and the retired `muted_threads` vary
    between runs with no behavioural difference, so hashing them would make every run incomparable
    with every other for no reason.
    """
    # Imported here rather than at module scope: `participation` pulls in the telemetry and config
    # graph, and this module is imported from the database layer.
    from message_processor.participation import resolve_participation_level

    mode = baseline["response_mode"] or getattr(config, "channel_response_mode", None)
    return {
        "response_mode": mode,
        "reply_in_channel": (baseline["reply_in_channel"]
                             if baseline["reply_in_channel"] is not None
                             else getattr(config, "reply_in_channel_default", None)),
        # No global `participation_level` exists — the level DERIVES from response_mode
        # (message_processor/participation.py:99), so the resolved value is what enters the hash.
        "participation_level": (baseline["participation_level"]
                                or resolve_participation_level({"response_mode": mode})),
        "snoozed_until": baseline["snoozed_until"],
        "model": baseline["model"] or getattr(config, "gpt_model", None),
        "reasoning_effort": (baseline["reasoning_effort"]
                             or getattr(config, "default_reasoning_effort", None)),
        "verbosity": baseline["verbosity"] or getattr(config, "default_verbosity", None),
        "ambient_memory": (baseline["ambient_memory"]
                           if baseline["ambient_memory"] is not None
                           else getattr(config, "enable_ambient_memory", None)),
    }


def classification_overlay_version(human_bot_ids) -> str:
    """sha256 of the sorted human-classified bot ids, first 16 chars.

    Separate from `fixture_version` because this is the one fixture member sourced from the live
    workspace rather than from configuration.
    """
    payload = json.dumps(sorted(human_bot_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_fixture(config: Any = None, human_bot_ids=(), now_ts: Optional[str] = None
                  ) -> EpochFixture:
    """The pinned overlay values for a battery, plus the two version stamps.

    `human_bot_ids` is the FROZEN UNION of the configured `DEV_TREAT_BOT_IDS_AS_HUMAN` carve-out
    and any second-party id the caller discovered (finding #27): a battery seeded through a user
    token would otherwise classify differently under the fence than outside it.

    LEASE-STABLE. The value is minted at activation and carried unchanged into every case, so it
    distinguishes one BATTERY from another rather than one moment from another within a battery.
    A second party that changed identity mid-run is NOT detected, and for a dev battery that is
    accepted — the alternative is a `bots.info` round-trip on every case.
    """
    if config is None:
        from config import BotConfig
        config = BotConfig()
    from config import CHANNEL_CAPABILITY_KEYS

    stamp = now_ts or f"{time.time():.6f}"
    configured = tuple(getattr(config, "dev_treat_bot_ids_as_human", None) or ())
    ids = tuple(sorted({str(b).strip() for b in (*configured, *human_bot_ids) if str(b).strip()}))

    baseline = _fixture_channel_settings(stamp)
    profile = _resolved_profile(config)
    payload = {
        "memory": [],
        "steering": EPOCH_STEERING_FIXTURE,
        "profile": {k: profile[k] for k in CHANNEL_CAPABILITY_KEYS},
        "settings": _resolved_settings(config, baseline),
        "human_bot_ids": list(ids),
    }
    version = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()
    return EpochFixture(
        memory=(),
        steering=EPOCH_STEERING_FIXTURE,
        channel_settings=baseline,
        human_bot_ids=ids,
        fixture_version=version,
        classification_overlay_version=classification_overlay_version(ids))


# --- the overlay store (§7.1d, §7.1h, §7.1h-bis) ---------------------------------------------


class EpochOverlayStore:
    """In-memory, per-(team, channel, lease, case). NEVER touches production rows.

    RESET AT EVERY SUB-EPOCH ADVANCE, not merely at teardown: keying it to the lease alone would
    let case N's remembered facts contaminate case N+1.

    Slim carries FIVE overlaid populations — memory, steering, channel settings, the window anchor
    and response feedback — which is §7.1h-bis's full OVERLAY column of the derived-state table.
    The catalog overlays (canvas, container) of §7.1i are not built: they exist to hide pre-fence
    durable assets from a fenced request, and hiding them needs read-side hooks in code this wave
    does not own.
    """

    def __init__(self, fixture: EpochFixture):
        self._fixture = fixture
        self._cases: Dict[FenceKey, Dict[str, Any]] = {}

    def _case(self, key: FenceKey) -> Dict[str, Any]:
        case = self._cases.get(key)
        if case is None:
            stamp = f"{time.time():.6f}"
            case = self._cases[key] = {
                "memory": [MemoryRow(id=i, content=r.content, scope=r.scope, author=r.author,
                                     created_ts=r.created_ts, updated_ts=r.updated_ts)
                           for i, r in enumerate(self._fixture.memory,
                                                 start=OVERLAY_MEMORY_ID_BASE + 1)],
                "next_id": OVERLAY_MEMORY_ID_BASE + 1 + len(self._fixture.memory),
                "steering": self._fixture.steering,
                "settings": dict(self._fixture.channel_settings),
                # INITIAL STATE None — a fenced channel starts with NO floor, so its first turn
                # takes the cold path and derives one. Inheriting production's floor would shape
                # the battery's first window from whatever the room was doing before it.
                "anchor": None,
                "feedback": {},
                "created_ts": stamp,
            }
        return case

    # --- channel memory ---------------------------------------------------------------
    def memory(self, key: FenceKey) -> List[Dict[str, Any]]:
        """The `get_channel_memory` shape: non-policy rows, oldest updated first."""
        rows = [r for r in self._case(key)["memory"] if r.scope != "policy"]
        rows.sort(key=lambda r: r.updated_ts)
        return [{"id": r.id, "channel_id": key.channel_id, "scope": r.scope,
                 "content": r.content, "author": r.author,
                 "created_ts": r.created_ts, "updated_ts": r.updated_ts} for r in rows]

    def add_memory(self, key: FenceKey, content: str, *, scope: str = "channel",
                   author: Optional[str] = None) -> int:
        case = self._case(key)
        stamp = f"{time.time():.6f}"
        row_id = case["next_id"]
        case["next_id"] += 1
        case["memory"].append(MemoryRow(id=row_id, content=content, scope=scope, author=author,
                                        created_ts=stamp, updated_ts=stamp))
        return row_id

    def update_memory(self, key: FenceKey, memory_id: int, content: str) -> bool:
        case = self._case(key)
        for i, row in enumerate(case["memory"]):
            if row.id == memory_id:
                case["memory"][i] = replace(row, content=content,
                                            updated_ts=f"{time.time():.6f}")
                return True
        return False

    def delete_memory(self, key: FenceKey, memory_id: int) -> bool:
        case = self._case(key)
        before = len(case["memory"])
        case["memory"] = [r for r in case["memory"] if r.id != memory_id]
        return len(case["memory"]) != before

    # --- standing policy / steering ---------------------------------------------------
    def steering(self, key: FenceKey) -> str:
        return self._case(key)["steering"]

    def steering_row(self, key: FenceKey) -> Optional[Dict[str, Any]]:
        """The `get_channel_policy` shape, or None when the operator cleared it."""
        text = (self._case(key)["steering"] or "").strip()
        if not text:
            return None
        stamp = self._case(key)["created_ts"]
        return {"id": OVERLAY_MEMORY_ID_BASE, "channel_id": key.channel_id,
                "scope": "policy", "content": text,
                "author": "epoch_fence", "created_ts": stamp, "updated_ts": stamp}

    def set_steering(self, key: FenceKey, content: Optional[str]) -> None:
        self._case(key)["steering"] = (content or "").strip()

    def set_steering_if_unchanged(self, key: FenceKey, content: Optional[str], *,
                                  expected_hash: Optional[str], hasher) -> bool:
        """Overlay CAS, comparing against the OVERLAY's hash rather than production's.

        `hasher` is the caller's `memory_content_hash`, passed in so this module does not import
        the production hashing helper for one call. Returns False on mismatch, exactly as
        `set_channel_policy_if_unchanged_async` does.
        """
        current = (self._case(key)["steering"] or "").strip()
        if (hasher(current) if current else "") != (expected_hash or ""):
            return False
        self.set_steering(key, content)
        return True

    # --- channel settings --------------------------------------------------------------
    def channel_settings(self, key: FenceKey) -> Dict[str, Any]:
        return dict(self._case(key)["settings"])

    def set_channel_settings(self, key: FenceKey, **fields: Any) -> None:
        """PARTIAL update with production's `_UNSET` semantics: only the fields PRESENT in the call
        are written, so a modal submission that omits a block preserves the overlay's value for
        it. The caller strips `_UNSET` before calling."""
        case = self._case(key)
        case["settings"].update(fields)
        case["settings"]["updated_ts"] = f"{time.time():.6f}"

    # --- the window anchor (SELECTION state) --------------------------------------------
    def window_anchor(self, key: FenceKey) -> Optional[Dict[str, Any]]:
        return dict(self._case(key)["anchor"]) if self._case(key)["anchor"] else None

    def advance_window_anchor(self, key: FenceKey, floor_ts: str,
                              selection_version: int) -> bool:
        """The same THREE version rules as production: a higher version overwrites, an equal
        version may only move the floor FORWARD, a lower version is REJECTED. True when the row
        moved — the `anchor_advanced` the carrier reports."""
        case = self._case(key)
        current = case["anchor"]
        incoming = {"floor_ts": str(floor_ts), "selection_version": int(selection_version)}
        if current is None:
            case["anchor"] = incoming
            return True
        if incoming["selection_version"] > current["selection_version"]:
            case["anchor"] = incoming
            return True
        if (incoming["selection_version"] == current["selection_version"]
                and parse_ts(incoming["floor_ts"]) > parse_ts(current["floor_ts"])):
            case["anchor"] = incoming
            return True
        return False

    # --- response feedback (PREFERENCE state) --------------------------------------------
    def add_feedback(self, key: FenceKey, row: FeedbackRow) -> None:
        """A 👍/👎 on a fenced surface. Upserted on production's `(message_ts, user_id, source)`
        conflict key; the production table is never touched."""
        self._case(key)["feedback"][(row.message_ts, row.user_id, row.source)] = row

    def delete_feedback(self, key: FenceKey, message_ts: str, user_id: str, source: str) -> None:
        self._case(key)["feedback"].pop((message_ts, user_id, source), None)

    def feedback(self, key: FenceKey) -> Tuple[FeedbackRow, ...]:
        return tuple(self._case(key)["feedback"].values())

    # --- lifecycle -----------------------------------------------------------------------
    def reset_subepoch(self, key: FenceKey) -> None:
        """Reclaim the OUTGOING case's entries. An advance orphans them by construction — every
        read and write is keyed by the current sub-epoch — so this merely frees the memory."""
        self._cases.pop(key, None)

    def drop(self) -> None:
        self._cases.clear()


def routed_key(fence: ActiveFence) -> FenceKey:
    """The overlay key THIS CALLER writes to — the caller's OWN sub-epoch, not the live one.

    A case-N turn that is still running when case N+1 opens is not stopped by slim (there is no
    drain), so it will keep reading and writing. Keying its accessors on the LIVE case would let
    its settings edits, remembered facts, feedback and anchor advances land inside N+1 — the exact
    contamination sub-epochs exist to prevent, arriving through the overlay instead of through
    Slack.

    So the key comes from the task's OWN stamp when it has one. `advance` has already reclaimed
    that sub-epoch's bucket, so a stale write allocates a fresh orphan bucket that no reader will
    ever ask for, and a stale read gets the fixture. It goes nowhere, which is the requirement —
    and it does so without raising out of a database accessor, where a raise would surface as a
    mystery failure far from the cause.

    An UNSTAMPED caller gets the live key. That is slim's accepted weakening, stated in the module
    docstring: without stamping at every dispatch site, treating absence of a stamp as staleness
    would send the battery's own Bolt callbacks into an orphan bucket.
    """
    ctx = current_epoch()
    if ctx is None or (ctx.lease_id, ctx.test_epoch_id) == (fence.context.lease_id,
                                                            fence.context.test_epoch_id):
        return fence.key
    return FenceKey(fence.team_id, fence.channel_id, ctx.lease_id,
                    ctx.test_epoch_id or ARMED_EPOCH)


def overlay_for_channel(channel_id: Optional[str], team_id: Optional[str] = None
                        ) -> Optional[Tuple[EpochOverlayStore, FenceKey]]:
    """The overlay a durable read or write for `channel_id` must be served from, or None.

    THE production no-op path is the first line: an empty registry is impossible without the flag,
    so an unfenced process reaches this, returns None, and runs identical code to today.
    """
    if not _FENCES:
        return None
    fence = active_fence(team_id, channel_id)
    if fence is None:
        return None
    return fence.overlay, routed_key(fence)


def overlay_by_memory_id(memory_id: Any) -> Optional[Tuple[EpochOverlayStore, FenceKey]]:
    """Route a channel-less `channel_memory` accessor by its row id.

    `update_channel_memory_async(memory_id, …)` and `delete_channel_memory_async(memory_id)` name
    a row and no channel, so a fenced modal reconcile could otherwise edit a PRODUCTION row that
    happened to share the overlay's small integer. An id below `OVERLAY_MEMORY_ID_BASE` can only
    be production and is never routed here.
    """
    if not _FENCES:
        return None
    try:
        row_id = int(memory_id)
    except (TypeError, ValueError):
        return None
    if row_id < OVERLAY_MEMORY_ID_BASE:
        # A PRODUCTION id. It may still name a row in a fenced channel — a row that predates the
        # fence — and rewriting that would break the promise that no production policy row moves
        # while a battery runs. The caller resolves the row's channel and refuses in that case;
        # here we only say "this is not an overlay row".
        return None
    for fence in _FENCES.values():
        key = routed_key(fence)
        if any(r["id"] == row_id for r in fence.overlay.memory(key)):
            return fence.overlay, key
    # An id in the overlay band with no matching row still belongs to the overlay, not to
    # production — returning the fence covering it keeps the write out of the real table.
    fence = next(iter(_FENCES.values()))
    return fence.overlay, routed_key(fence)


def overlay_by_feedback(message_ts: str, user_id: str, source: str
                        ) -> Optional[Tuple[EpochOverlayStore, FenceKey]]:
    """Route `delete_response_feedback_async`, which names a message and no channel."""
    if not _FENCES:
        return None
    for fence in _FENCES.values():
        key = routed_key(fence)
        if any((r.message_ts, r.user_id, r.source) == (message_ts, user_id, source)
               for r in fence.overlay.feedback(key)):
            return fence.overlay, key
    return None


def any_fence_installed() -> bool:
    """True when ANY fence is up. The cheap first test for the channel-less accessors, which must
    resolve their row's channel before they can tell a production write from a fenced one."""
    return bool(_FENCES)


def _reset_all() -> None:
    """Drop every fence and the allowlist. For the harness demo and for a watcher restart — NOT a
    production path; nothing calls it while a bot is serving."""
    global _SCOPES
    for team, channel in list(_FENCES):
        remove_fence(team, channel)
    _SCOPES = None
