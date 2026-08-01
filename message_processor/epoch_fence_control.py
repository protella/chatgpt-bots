"""The epoch fence's control surface and watcher — dev only, SLIM build.

Imported ONLY when `DEV_EPOCH_FENCE_ENABLE` is set. `main.py` starts the watcher behind that flag
and nothing else in the process references this module, so without the flag no control file is
opened, no fence table is created, and no watcher task exists.

THE CONTROL FILE is `<DEV_EPOCH_FENCE_DIR>/epoch_fence.json`, the same shape the dev turn barriers
already use for their control directory. Its ABSENCE means no fence is requested; a file that is
present but UNREADABLE never means that — a corrupted file mid-battery would otherwise silently
unfence a live channel, which is the failure the whole mechanism exists to prevent.

AUTHORIZATION differs by command. `activate` carries no token, because there is no token to present
until activation mints one; its authorization is the control surface itself — an owner-only `0700`
directory the bot is reading, plus the hardcoded tenant-scoped allowlist. `advance` and `release`
must present the `owner_token` activation minted.

`command_id` IS WHAT MAKES A 0.1s POLLER SAFE. A watcher that merely read `command` would re-apply
the same `advance` ten times a second. The watcher applies a command only when
`command_id > acked_id`, and the ACK is the watcher rewriting the file with `acked_id = command_id`.
That is exactly-once in the steady state and inspectable, but it is NOT exactly-once across a crash
between the DB transition and the ack — two stores, two writes. Reconciliation closes that: the
lease ROW is the truth, and every replay decides from it.
"""
from __future__ import annotations

import json
import os
import secrets
import stat
import time
from typing import Any, Dict, Optional, Tuple

from logger import setup_logger

from message_processor import epoch_fence

logger = setup_logger(name="slack_bot.EpochFenceControl")

CONTROL_FILENAME = "epoch_fence.json"
DEFAULT_CONTROL_DIR = "data/epoch_fence"

COMMANDS = ("activate", "advance", "release")

#: Commands that must present the lease's `owner_token`. `activate` is deliberately absent:
#: requiring a token there would reject every valid activation (finding #25).
TOKEN_REQUIRED = ("advance", "release")


class ControlFileError(Exception):
    """The control file is present but cannot be honoured. NEVER read as 'no fence requested'."""


def control_dir() -> str:
    """The control directory. Its own subdirectory by default, NOT bare `data/`.

    The file carries a bearer token, so the directory must be owner-only — and a directory cannot
    be mode `0600` and remain traversable, which is what the spec's earlier text asked for
    (finding #16). `0700` on a dedicated directory is the shape that actually works, and putting
    it beside `data/` rather than inside it keeps the bot's own data directory at its usual mode.
    """
    return (os.environ.get(epoch_fence.ENV_CONTROL_DIR) or "").strip() or DEFAULT_CONTROL_DIR


def control_path() -> str:
    return os.path.join(control_dir(), CONTROL_FILENAME)


def ensure_control_dir() -> str:
    """Create the control directory `0700` and refuse loudly if it is not owner-only.

    Activation has NO token to present, so this directory IS the authorization. A group- or
    world-writable one would let any local user activate a fence, and a group- or world-READABLE
    one would leak the bearer token the watcher writes back into it.
    """
    path = control_dir()
    os.makedirs(path, mode=0o700, exist_ok=True)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode & 0o077:
        raise ControlFileError(
            f"epoch fence control directory {path!r} is mode {mode:04o}; it must be 0700 — it "
            f"holds the lease bearer token and is the only thing authorizing an activation")
    if os.stat(path).st_uid != os.getuid():
        raise ControlFileError(
            f"epoch fence control directory {path!r} is not owned by uid {os.getuid()}")
    return path


def write_control(payload: Dict[str, Any]) -> None:
    """Atomic, private write. Both the CLI and the watcher use this, in both directions.

    Mode `0o600` is set on the TEMP file BEFORE the replace: the file carries a bearer token, so
    it must never be briefly world-readable, and a poller reading ten times a second must never
    see it half-written.
    """
    directory = ensure_control_dir()
    path = os.path.join(directory, CONTROL_FILENAME)
    tmp = f"{path}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_control() -> Optional[Dict[str, Any]]:
    """The control file, or None when it is ABSENT. Raises `ControlFileError` when it is present
    and malformed — a truncated read is not an absence."""
    path = control_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ControlFileError(f"epoch fence control file {path!r} could not be read: {e}") from e
    if not raw.strip():
        raise ControlFileError(f"epoch fence control file {path!r} is empty or truncated")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ControlFileError(f"epoch fence control file {path!r} is not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ControlFileError(f"epoch fence control file {path!r} is not a JSON object")
    return payload


def validate_command(payload: Dict[str, Any], acked_id: int) -> Optional[Dict[str, Any]]:
    """The parsed command, or None when there is nothing to apply.

    Raises `ControlFileError` for anything malformed — an unknown `command`, a token missing on
    `advance`/`release`, a `start_ts` supplied by the CLI, a backward `command_id`. Every one of
    those leaves an active fence exactly as it was and is NOT acked, so an operator who fixes the
    file retries the same `command_id`.
    """
    raw_id = payload.get("command_id")
    if not isinstance(raw_id, int) or isinstance(raw_id, bool):
        raise ControlFileError(f"command_id must be an integer, got {raw_id!r}")
    if raw_id < acked_id:
        raise ControlFileError(
            f"command_id went BACKWARD ({raw_id} < acked {acked_id}); the file is malformed")
    if raw_id <= acked_id:
        # The steady state between commands, and what the watcher sees on all but one poll.
        return None

    command = payload.get("command")
    if command not in COMMANDS:
        raise ControlFileError(f"unknown command {command!r}; expected one of {COMMANDS}")
    if command in TOKEN_REQUIRED and not (payload.get("owner_token") or "").strip():
        raise ControlFileError(f"{command} requires the owner_token activation minted")
    if command == "advance" and not (payload.get("test_epoch_id") or "").strip():
        raise ControlFileError("advance requires a test_epoch_id")
    if payload.get("start_ts") is not None:
        # The CLI must never choose the boundary: a boundary chosen when the command was written
        # predates the quiescence that follows it (finding #23).
        raise ControlFileError(
            "start_ts is chosen by the watcher, never supplied by the CLI; the file names one")
    return {"command_id": raw_id, "command": command,
            "team_id": (payload.get("team_id") or "").strip() or None,
            "channel_id": (payload.get("channel_id") or "").strip(),
            "owner_token": (payload.get("owner_token") or "").strip() or None,
            "test_epoch_id": (payload.get("test_epoch_id") or "").strip() or None}


def _now() -> str:
    return f"{time.time():.6f}"


def _passed(expiry_ts: Optional[str]) -> bool:
    """True when an expiry has already gone by. An unparseable or absent one counts as passed —
    a lease whose lifetime cannot be read is not a lease anything should resume."""
    try:
        return time.time() >= float(expiry_ts)
    except (TypeError, ValueError):
        return True


class EpochFenceWatcher:
    """Polls the control file and the lease row, and owns every fence transition.

    Started only under `DEV_EPOCH_FENCE_ENABLE`, after the outbound-receipts service and before the
    coverage bootstrap, so no channel work runs against an unstamped fence. Cancelled and awaited
    FIRST at shutdown, ahead of `client.stop()`.
    """

    #: How long boot will wait for initialisation — identity, schema, invalidation — to land.
    #: Generous, because what it is waiting for is the deny-only fence over an interrupted
    #: battery, and beginning service before that exists is the failure it prevents.
    READY_TIMEOUT_SECONDS = 60.0

    def __init__(self, client: Any):
        import asyncio
        self._client = client
        self._db = getattr(client, "db", None)
        self._task = None
        self._ready = False
        #: Set when `_boot` has finished, SUCCEEDED OR NOT — it is a "startup is over" edge, not a
        #: success flag. `wait_ready` reports which it was via `self._ready`.
        self._ready_event = asyncio.Event()
        #: The tokens THIS process minted. A same-process replay of an un-acked activation can
        #: resume from these; a process RESTART cannot, and boot invalidation is what handles that
        #: case instead (finding #15).
        self._minted: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # --- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        import asyncio
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="epoch_fence_watcher")
        self._task.add_done_callback(
            lambda t: t.cancelled() or (t.exception() and logger.error(
                f"Epoch fence watcher stopped with: {t.exception()}")))

    async def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until startup has finished. True when it SUCCEEDED.

        The caller is `main.initialize`, and the point is ordering: boot invalidation and its
        deny-only fence must be installed before the bot starts serving, or a channel whose
        battery a restart interrupted answers normally in the gap.
        """
        import asyncio
        try:
            await asyncio.wait_for(self._ready_event.wait(),
                                   timeout if timeout is not None else self.READY_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return False
        return self._ready

    async def stop(self) -> None:
        import asyncio
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Epoch fence watcher shutdown error: {e}")
        self._task = None

    # --- the poll loop ------------------------------------------------------------------
    async def _run(self) -> None:
        import asyncio

        # Read DYNAMICALLY at each use, never `from … import _POLL_SECONDS`: a from-import copies
        # the binding, so a monkeypatch of the module attribute would not reach this loop.
        from message_processor import dev_barriers

        try:
            await self._boot()
        finally:
            # The edge fires whether startup succeeded or failed. A boot that cannot resolve the
            # workspace must not leave `main` waiting out the full timeout for news it already has.
            self._ready_event.set()
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — a bad tick must not kill the watcher
                logger.error(f"Epoch fence watcher tick failed: {e}", exc_info=True)
            await asyncio.sleep(dev_barriers._POLL_SECONDS)

    async def _boot(self) -> None:
        """Resolve the workspace, build the allowlist, create the fence table, and INVALIDATE any
        battery that a restart interrupted.

        The lease row is durable; the fixture, the overlays and the runtime state are NOT. So a
        restart mid-battery finds a live, unexpired lease and no way to be the process that owned
        it. Resuming would mean reconstructing overlay CONTENTS that no durable record holds;
        waiting for natural expiry would leave the channel answering normally, unfenced, in the
        meantime. **A battery that survived a bot restart is not comparable to one that did not**,
        so the only honest outcome is to say so and stop.
        """
        import asyncio

        team = None
        for _ in range(100):
            ensure = getattr(self._client, "_ensure_self_identity", None)
            if ensure is not None:
                try:
                    await ensure()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Epoch fence: identity probe failed: {e}")
            team = getattr(self._client, "self_team_id", None)
            if team:
                break
            await asyncio.sleep(0.5)
        if not team:
            logger.error(
                "Epoch fence watcher could not resolve the workspace team id from auth.test; "
                "no scope is allowlisted and every fence request will refuse.")
            return

        epoch_fence.init_epoch_scopes(team)
        await self._db.ensure_epoch_fence_schema_async()
        stranded = await self._db.invalidate_stale_epoch_leases_async(_now())
        for row in stranded:
            # The DURABLE mark is only half of it. `authorize_effect` reads the RUNTIME registry,
            # and an empty registry authorizes everything — so without this the channel would go
            # right on answering, unfenced, and the ERROR below would be a promise the process
            # does not keep.
            epoch_fence.install_denied_fence(
                row["team_id"], row["channel_id"], row["lease_id"], row["expiry_ts"])
            logger.error(
                f"Epoch fence INVALIDATED by restart: channel={row['channel_id']} "
                f"lease={row['lease_id']} case={row['test_epoch_id']} — the battery's overlays "
                f"did not survive the restart, so its results are not comparable. The scope is "
                f"shut and stays shut until an explicit `epoch_fence.py release`.")
        try:
            ensure_control_dir()
        except ControlFileError as e:
            logger.error(f"Epoch fence control directory refused: {e}")
            return
        self._ready = True

    async def _tick(self) -> None:
        if not self._ready:
            return
        await self._check_expiry()
        try:
            payload = read_control()
        except ControlFileError as e:
            logger.error(f"Epoch fence control file REFUSED, nothing changed: {e}")
            return
        if payload is None:
            return
        acked = payload.get("acked_id")
        acked = acked if isinstance(acked, int) and not isinstance(acked, bool) else 0
        try:
            command = validate_command(payload, acked)
        except ControlFileError as e:
            logger.error(f"Epoch fence command REFUSED, nothing changed: {e}")
            return
        if command is None:
            return
        try:
            ack = await self._apply(command)
        except epoch_fence.EpochFenceError as e:
            logger.error(f"Epoch fence command {command['command']!r} REFUSED: {e}")
            return
        if ack is None:
            return
        # THE ACK IS THE HARNESS'S SIGNAL. A no-op replay still acks — withholding it because the
        # work was already done would hang the battery on a command that succeeded.
        write_control({**payload, "acked_id": command["command_id"], **ack})

    # --- transitions ---------------------------------------------------------------------
    async def _apply(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        scope = epoch_fence.require_scope(
            command["team_id"] or getattr(self._client, "self_team_id", None),
            command["channel_id"])
        handler = {"activate": self._activate, "advance": self._advance,
                   "release": self._release}[command["command"]]
        return await handler(scope, command)

    async def _activate(self, scope, command) -> Optional[Dict[str, Any]]:
        team, channel = scope
        now = _now()
        lease_id = f"lease-{secrets.token_hex(6)}"
        owner_token = secrets.token_urlsafe(24)
        expiry = f"{float(now) + epoch_fence.EXPIRY_SECONDS:.6f}"
        won = await self._db.acquire_epoch_lease_async(
            team, channel, lease_id, owner_token, now, expiry, command["command_id"])
        if won:
            # RECORDED IMMEDIATELY, before the fixture is built or the fence installed. Everything
            # after the acquire can fail — a fixture that raises, an install that does not run —
            # and if the token were only recorded at the end, the retry would find a live row it
            # could neither acquire nor recognise as its own, and the activation would be stranded
            # for good.
            self._minted[(team, channel)] = {"lease_id": lease_id, "owner_token": owner_token}
        else:
            row = await self._db.read_epoch_lease_async(team, channel)
            # THE ROW RECOGNISES ITS OWN WORK, through `applied_command_id` — written in the SAME
            # statement as the acquire, so "the row exists and this command made it" is one atomic
            # fact rather than an inference from process-local bookkeeping that a failure may have
            # skipped. A row from a genuinely earlier process cannot reach here: boot invalidation
            # runs first and `invalidated` is not one of the resumable states.
            if (row and row["applied_command_id"] == command["command_id"]
                    and row["state"] in ("armed", "active", "closing")
                    and not _passed(row["expiry_ts"])):
                lease_id, owner_token, expiry = (row["lease_id"], row["owner_token"],
                                                 row["expiry_ts"])
                self._minted[(team, channel)] = {"lease_id": lease_id,
                                                 "owner_token": owner_token}
                logger.warning(
                    f"Epoch fence activation RESUMED for {channel} lease={lease_id} — the acquire "
                    f"had landed and the ack had not; re-deriving the fixture and acking with the "
                    f"token read back from the row.")
            else:
                raise epoch_fence.EpochScopeError(
                    f"a live epoch lease already holds {channel} "
                    f"(state={row and row['state']}, lease={row and row['lease_id']}); "
                    f"release it before activating another battery")

        fixture = epoch_fence.build_fixture(
            human_bot_ids=(getattr(self._client, "bot_id", None),) if
            getattr(self._client, "bot_id", None) else (),
            now_ts=now)
        epoch_fence.install_fence(epoch_fence.ActiveFence(
            team_id=team, channel_id=channel,
            context=epoch_fence.EpochContext(
                lease_id=lease_id, test_epoch_id=None, start_ts=None,
                fixture_version=fixture.fixture_version,
                classification_overlay_version=fixture.classification_overlay_version),
            fixture=fixture, overlay=epoch_fence.EpochOverlayStore(fixture),
            state="armed", expiry_ts=expiry))
        return {"lease_id": lease_id, "owner_token": owner_token, "state": "armed",
                "team_id": team, "expiry_ts": expiry,
                "fixture_version": fixture.fixture_version,
                "classification_overlay_version": fixture.classification_overlay_version}

    async def _advance(self, scope, command) -> Optional[Dict[str, Any]]:
        team, channel = scope
        row = await self._db.read_epoch_lease_async(team, channel)
        if row is None or row["owner_token"] != command["owner_token"]:
            raise epoch_fence.EpochScopeError(
                f"advance presented the wrong owner_token for {channel}")
        # Checked BEFORE the durable write, not after: moving the row and then discovering there
        # is no registry to move with it is how the two stores disagree in the first place.
        live = epoch_fence.active_fence(team, channel)
        if row["test_epoch_id"] != command["test_epoch_id"] and (
                live is None or live.state == "invalidated"):
            raise epoch_fence.EpochScopeError(
                f"advance refused for {channel}: this process holds no live fence to open a case "
                f"in. Release and re-activate.")
        if row["test_epoch_id"] == command["test_epoch_id"]:
            # The DURABLE half already landed; the ack was lost. But the transition is TWO writes
            # to two stores, and SQLite agreeing is not evidence that the registry moved — an ack
            # here on its own would tell the harness a case is open while the overlays and the
            # effect check are still sitting in the previous one. So reconcile the runtime to the
            # row FIRST, and only then ack.
            fence = epoch_fence.active_fence(team, channel)
            if fence is None or fence.state == "invalidated":
                raise epoch_fence.EpochScopeError(
                    f"advance cannot be replayed for {channel}: the durable row says case "
                    f"{row['test_epoch_id']!r} is open, but this process holds no live fence to "
                    f"open it in. Release and re-activate — the overlays are gone.")
            if fence.context.test_epoch_id != row["test_epoch_id"]:
                epoch_fence.advance_fence(team, channel, row["test_epoch_id"], row["start_ts"],
                                          row["expiry_ts"])
                logger.warning(
                    f"Epoch fence advance RECONCILED for {channel} case={row['test_epoch_id']} — "
                    f"the durable transition had landed and the registry had not.")
            return {"state": "active", "start_ts": row["start_ts"],
                    "test_epoch_id": row["test_epoch_id"]}

        # THE WATCHER CHOOSES start_ts, HERE, at the moment the case opens. Slim has no drain to
        # wait out, so this is simply the latest possible boundary: anything Slack accepted before
        # this instant belongs to the previous case.
        start_ts = _now()
        expiry = f"{float(start_ts) + epoch_fence.EXPIRY_SECONDS:.6f}"
        moved = await self._db.advance_epoch_subepoch_async(
            team, channel, command["owner_token"], command["test_epoch_id"], start_ts,
            expiry, command["command_id"])
        if not moved:
            raise epoch_fence.EpochScopeError(
                f"advance refused for {channel}: the lease row is not acquirable by this token")
        epoch_fence.advance_fence(team, channel, command["test_epoch_id"], start_ts, expiry)
        return {"state": "active", "start_ts": start_ts,
                "test_epoch_id": command["test_epoch_id"]}

    async def _release(self, scope, command) -> Optional[Dict[str, Any]]:
        team, channel = scope
        row = await self._db.read_epoch_lease_async(team, channel)
        if row is None or row["state"] == "released":
            epoch_fence.remove_fence(team, channel)
            return {"state": "released"}
        if row["owner_token"] != command["owner_token"]:
            raise epoch_fence.EpochScopeError(
                f"release presented the wrong owner_token for {channel}")
        epoch_fence.set_fence_state(team, channel, "closing")
        await self._db.release_epoch_lease_async(
            team, channel, command["owner_token"], command["command_id"])
        epoch_fence.remove_fence(team, channel)
        self._minted.pop((team, channel), None)
        return {"state": "released"}

    async def _check_expiry(self) -> None:
        """A fence never outlives its harness by more than one expiry window.

        The CLI harness owns the heartbeat; the bot must NOT heartbeat its own fence, or an
        abandoned battery would keep the channel shut forever. Expiry is the dead-man's switch.

        THE ROW IS WHERE THE HEARTBEAT LANDS, so the row is what this reads. The harness bumps
        `expiry_ts` in SQLite every 30 seconds; the `ActiveFence` in memory is frozen at the value
        its last transition wrote. Judging expiry from the frozen copy tore down live batteries at
        the 300-second mark while the harness was faithfully keeping them alive — so each tick
        refreshes the runtime expiry from the row first, and `authorize_effect` (which reads the
        runtime copy) is then never more than one 0.1s poll behind the truth.
        """
        for (team, channel), fence in list(epoch_fence._FENCES.items()):
            # A restart-invalidated scope has no harness and no heartbeat by definition. It is
            # deny-only and must STAY installed — expiring it would silently reopen the channel,
            # which is precisely what invalidation exists to prevent. Only a release clears it.
            if fence.state == "invalidated":
                continue
            row = await self._db.read_epoch_lease_async(team, channel)
            if row is not None and row["expiry_ts"] != fence.expiry_ts:
                epoch_fence.set_fence_expiry(team, channel, row["expiry_ts"])
                fence = epoch_fence.active_fence(team, channel)
            if not epoch_fence._expired(fence):
                continue
            logger.error(
                f"Epoch fence EXPIRED for {channel} lease={fence.context.lease_id} "
                f"case={fence.context.test_epoch_id} — the harness stopped heartbeating. "
                f"Dropping the overlays and releasing the lease.")
            mine = self._minted.get((team, channel))
            if mine is not None:
                try:
                    await self._db.release_epoch_lease_async(
                        team, channel, mine["owner_token"], None)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Epoch fence expiry release failed for {channel}: {e}")
                self._minted.pop((team, channel), None)
            epoch_fence.remove_fence(team, channel)
