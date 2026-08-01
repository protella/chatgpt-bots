#!/usr/bin/env python3
"""Drive the dev-only epoch fence from the live battery harness.

    tools/epoch_fence.py activate  --channel C0BKX77NU66
    tools/epoch_fence.py advance   --channel C0BKX77NU66 --test-epoch-id case-01
    tools/epoch_fence.py heartbeat --channel C0BKX77NU66      # runs until interrupted
    tools/epoch_fence.py status    --channel C0BKX77NU66
    tools/epoch_fence.py release   --channel C0BKX77NU66

Every command except `status` and `heartbeat` writes the control file and waits for the bot's
watcher to ACK it. `status` writes nothing.

THE HARNESS OWNS THE HEARTBEAT, at a 30-second cadence, for the battery's duration — which is why
`heartbeat` is a long-running foreground process rather than a one-shot. **The bot must NOT
heartbeat its own fence**: an abandoned battery would then hold the channel forever and the expiry
could never fire, and the expiry is precisely the dead-man's switch for a harness that died.

`advance` deliberately takes NO `--start-ts`. The watcher chooses the case boundary at the moment
it opens the case; a boundary chosen when the command was written would be older than the command's
own application, and messages arriving in between would land in the wrong case. The chosen value
comes back in the ack and is printed here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from message_processor import epoch_fence  # noqa: E402
from message_processor import epoch_fence_control as control  # noqa: E402

ACK_TIMEOUT_SECONDS = 30.0
TOKEN_FILE = "epoch_fence_token.json"


def _token_path() -> str:
    return os.path.join(control.control_dir(), TOKEN_FILE)


def _save_token(channel_id: str, team_id: str, lease_id: str, owner_token: str) -> None:
    """Stash the token activation minted, so `advance`/`release`/`heartbeat` need no flags.

    Same directory, same `0600` write: it is the same secret as the control file's.
    """
    directory = control.ensure_control_dir()
    path = os.path.join(directory, TOKEN_FILE)
    tmp = f"{path}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"channel_id": channel_id, "team_id": team_id,
                   "lease_id": lease_id, "owner_token": owner_token}, handle)
    os.replace(tmp, path)


def _load_token(channel_id: str) -> Dict[str, Any]:
    try:
        with open(_token_path(), "r", encoding="utf-8") as handle:
            stored = json.load(handle)
    except FileNotFoundError:
        raise SystemExit(
            f"no stored lease for {channel_id}; run `activate` first (or the harness lost "
            f"{_token_path()})")
    if stored.get("channel_id") != channel_id:
        raise SystemExit(
            f"the stored lease is for {stored.get('channel_id')}, not {channel_id}")
    return stored


def _read_state() -> Dict[str, Any]:
    try:
        return control.read_control() or {}
    except control.ControlFileError:
        # A malformed file is still evidence of the last command_id we used; treat it as absent
        # for numbering and let the watcher keep refusing it until it is fixed.
        return {}


def _submit(command: str, channel_id: str, **fields: Any) -> Dict[str, Any]:
    """Write the next command and block until the watcher acks it."""
    existing = _read_state()
    acked = existing.get("acked_id") if isinstance(existing.get("acked_id"), int) else 0
    previous = existing.get("command_id") if isinstance(existing.get("command_id"), int) else 0
    command_id = max(acked, previous) + 1
    payload = {"command_id": command_id, "acked_id": acked, "command": command,
               "channel_id": channel_id, "start_ts": None, **fields}
    control.write_control(payload)

    deadline = time.monotonic() + ACK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        time.sleep(0.1)
        current = _read_state()
        if current.get("acked_id") == command_id:
            return current
    raise SystemExit(
        f"the bot did not ack {command} (command_id={command_id}) within "
        f"{ACK_TIMEOUT_SECONDS:.0f}s. Is it running with DEV_EPOCH_FENCE_ENABLE set, and is "
        f"{control.control_path()} the directory it is watching? Check the bot log — a refused "
        f"command logs at ERROR and is deliberately NOT acked, so the same command_id can be "
        f"retried once the cause is fixed.")


def cmd_activate(args) -> int:
    ack = _submit("activate", args.channel)
    _save_token(args.channel, ack.get("team_id") or "", ack["lease_id"], ack["owner_token"])
    print(f"activated  channel={args.channel}")
    print(f"  lease_id                       {ack['lease_id']}")
    print(f"  owner_token                    {ack['owner_token']}")
    print(f"  state                          {ack['state']} (no case open — every channel effect "
          f"is refused until the first advance)")
    print(f"  fixture_version                {ack.get('fixture_version')}")
    print(f"  classification_overlay_version {ack.get('classification_overlay_version')}")
    print(f"  expiry_ts                      {ack.get('expiry_ts')}  "
          f"(run `heartbeat` now, or the fence drops in "
          f"{epoch_fence.EXPIRY_SECONDS:.0f}s)")
    return 0


def cmd_advance(args) -> int:
    stored = _load_token(args.channel)
    ack = _submit("advance", args.channel, owner_token=stored["owner_token"],
                  lease_id=stored["lease_id"], test_epoch_id=args.test_epoch_id)
    print(f"advanced   channel={args.channel} case={ack.get('test_epoch_id')}")
    print(f"  start_ts   {ack.get('start_ts')}   (chosen by the watcher when the case opened)")
    return 0


def cmd_release(args) -> int:
    stored = _load_token(args.channel)
    _submit("release", args.channel, owner_token=stored["owner_token"],
            lease_id=stored["lease_id"])
    for path in (control.control_path(), _token_path()):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    print(f"released   channel={args.channel}; overlays dropped and the control file removed")
    return 0


def cmd_status(args) -> int:
    """Read the lease row and the control file. WRITES NOTHING."""
    stored: Optional[Dict[str, Any]]
    try:
        stored = _load_token(args.channel)
    except SystemExit:
        stored = None
    state = _read_state()
    print(f"control file  {control.control_path()}")
    print(f"  command_id  {state.get('command_id')}   acked_id {state.get('acked_id')}")
    print(f"  command     {state.get('command')}")
    if stored is None:
        print("stored lease  (none — this harness has not activated a fence)")
        return 0
    row = asyncio.run(_read_row(stored["team_id"], args.channel))
    print(f"stored lease  {stored['lease_id']}")
    if row is None:
        print("lease row     (absent — the bot has no record of this battery)")
        return 0
    now = time.time()
    left = float(row["expiry_ts"]) - now
    print(f"lease row     state={row['state']} case={row['test_epoch_id']} "
          f"start_ts={row['start_ts']}")
    print(f"              expiry_ts={row['expiry_ts']} ({left:+.0f}s from now)")
    return 0


async def _read_row(team_id: str, channel_id: str):
    from database import DatabaseManager
    db = DatabaseManager()
    try:
        await db.ensure_epoch_fence_schema_async()
        return await db.read_epoch_lease_async(team_id, channel_id)
    finally:
        db.close()


def cmd_heartbeat(args) -> int:
    """Hold the lease alive for the battery's duration. Foreground; Ctrl-C to stop.

    Writes DIRECTLY to the lease row rather than through the control file: a heartbeat is not a
    state transition, it carries no decision the watcher needs to reconcile, and routing
    one every 30 seconds through the command channel would burn command ids the operator reads.
    """
    stored = _load_token(args.channel)
    print(f"heartbeat  channel={args.channel} every {epoch_fence.HEARTBEAT_SECONDS:.0f}s "
          f"(Ctrl-C to stop; the fence then expires within "
          f"{epoch_fence.EXPIRY_SECONDS:.0f}s)")
    try:
        asyncio.run(_heartbeat_loop(stored["team_id"], args.channel, stored["owner_token"]))
    except KeyboardInterrupt:
        print("\nheartbeat stopped — the fence will expire on its own unless you `release`.")
    return 0


async def _heartbeat_loop(team_id: str, channel_id: str, owner_token: str) -> None:
    from database import DatabaseManager
    db = DatabaseManager()
    try:
        while True:
            now = f"{time.time():.6f}"
            expiry = f"{time.time() + epoch_fence.EXPIRY_SECONDS:.6f}"
            ok = await db.heartbeat_epoch_lease_async(team_id, channel_id, owner_token,
                                                      now, expiry)
            if not ok:
                # It does NOT revive an expired lease — that is the point of the expiry. The
                # battery has to re-acquire, and it must know that rather than keep beating.
                print("heartbeat REFUSED — the lease is expired, released or held by another "
                      "token. Re-activate before running more cases.", file=sys.stderr)
                return
            await asyncio.sleep(epoch_fence.HEARTBEAT_SECONDS)
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)

    activate = subs.add_parser("activate", help="claim the channel and mint a lease")
    activate.add_argument("--channel", required=True)
    activate.set_defaults(func=cmd_activate)

    advance = subs.add_parser("advance", help="open the next case")
    advance.add_argument("--channel", required=True)
    advance.add_argument("--test-epoch-id", required=True,
                         help="the case id, e.g. case-01. NOTE: there is no --start-ts; the "
                              "watcher chooses the boundary and reports it in the ack.")
    advance.set_defaults(func=cmd_advance)

    release = subs.add_parser("release", help="drop the overlays and hand the channel back")
    release.add_argument("--channel", required=True)
    release.set_defaults(func=cmd_release)

    status = subs.add_parser("status", help="read the lease row and control file; writes nothing")
    status.add_argument("--channel", required=True)
    status.set_defaults(func=cmd_status)

    heartbeat = subs.add_parser("heartbeat", help="hold the lease alive (foreground)")
    heartbeat.add_argument("--channel", required=True)
    heartbeat.set_defaults(func=cmd_heartbeat)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not epoch_fence.fence_enabled():
        print(f"{epoch_fence.ENV_ENABLE} is not set. The bot ignores the control file entirely "
              f"without it — set it in the DEV environment and restart the dev bot first.",
              file=sys.stderr)
        return 2
    if args.channel not in epoch_fence.FENCEABLE_CHANNELS:
        print(f"{args.channel} is not fenceable. The allowlist is hardcoded: "
              f"{', '.join(epoch_fence.FENCEABLE_CHANNELS)}.", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
