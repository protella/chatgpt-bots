#!/usr/bin/env python3
"""A READ-ONLY DIAGNOSTIC REPLAY of one channel's stream build (SHALLOW_STREAM_RESPEC §4.11).

    python3 -m tools.stream_probe --channel C0BKX77NU66 --origin <thread_ts> [--out report.json]
    python3 -m tools.stream_probe --channel C0BKX77NU66 --origin <A> --origin <B>

WHY THIS IS NOT A TURN. A real turn proceeds to the responder and may post into a production
channel, so it cannot be used to inspect a build. And a separate process cannot borrow the turn
machinery either: the admission watermark is PROCESS-LOCAL state, so a fresh CLI has no watermark
to pin and no frontier to drain. This does neither — it constructs its own `PreparedTurn` from a
wall-clock H and `frontier=0`, and calls the build PHASES directly.

`--origin` MAY BE GIVEN TWICE, and that is the render-equality mode: ONE periphery fetch, ONE H,
ONE shared sidecar read, then TWO immutable pins serialized in-process. Two separate invocations
cannot prove the same thing — their H values differ by construction, so their peripheries differ
and any hash comparison between them is meaningless.

ZERO DURABLE WRITES — Slack, DB and ledger. The scope is exact rather than absolute:

  * NO SLACK WRITES. The READ inventory is: `auth.test` (identity, once, via
    `_ensure_self_identity`), `conversations.history`, `conversations.replies`, and `users.info`
    when actor resolution reaches its remote pass. All four are reads; none of them writes. An
    earlier version of this docstring claimed only two, which was a promise the tool's own fake
    client happened to satisfy and the real client did not.
  * TWO OF THE TURN'S THREE DURABLE OUTPUTS ARE UNREACHABLE rather than suppressed: the anchor
    persist and the telemetry emit belong to `build_channel_stream`, which is never called. The
    third — the dirty compare-and-clear — lives in the phase this DOES call, and `probe=True`
    suppresses it. A diagnostic that silently un-dirtied roots would change the very next real
    turn's fetch plan.
  * NO LEDGER WRITES. `participation.jsonl` is untouched: the probe emits no telemetry at all,
    and deliberately invents no `kind` for itself — `turn_outcome.kind` shares one frozenset with
    `visible_action.kind` so a turn and its gate attempt stay comparable, and adding a probe kind
    there would silently change every historical participation denominator for a diagnostic.

TWO PROCESS-LOCAL MUTATIONS ARE NAMED HERE RATHER THAN LEFT TO BE DISCOVERED: resolving actor
names populates the client's in-memory username cache, and the actor tail would be another —
except that it is step 14, which belongs to the composer and is unreachable. Both die with the
process.

THE ONE FILE IT WRITES is its JSON report, which is its entire output.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import config
from database import DatabaseManager
from message_processor.channel_stream import (SELECTION_VERSION, PreparedTurn, build_channel_pin,
                                              build_origin_pin, eligible_for_stream,
                                              fetch_origin_thread, serialize_stream,
                                              serializer_config_snapshot)
from message_processor.utilities import reach_tools_for
from slack_client import actor_tail as actor_tail_module
from slack_client.history_fetch import FetchBudget, page_messages
from slack_client.normalizer import ORIGIN_REPLIES

PROBE_VERSION = 1


def _ts_sha256(timestamps: Sequence[str]) -> str:
    """A membership digest over SORTED ts values, newline-joined.

    MEMBERSHIP, NOT COUNTS. A build that dropped one message and gained another scores 120 = 120
    against an independent walk and is wrong; the hashes are what decide, and the counts are
    reported beside them only for diagnosis.
    """
    return hashlib.sha256("\n".join(sorted(timestamps)).encode("utf-8")).hexdigest()


async def _independent_origin_walk(client: Any, db: Any, *, team_id: str, channel_id: str,
                                   origin_root_ts: str, h: str, deadline_at: float,
                                   chrome_markers: Sequence[str]) -> Tuple[List[str], str]:
    """Page the origin thread AGAIN, independently, and return its ELIGIBLE ts list.

    The build's own `origin_count` cannot prove the origin was complete — it is the number under
    test comparing against nothing. This walks Slack a second time and compares membership.

    "Apply the same eligibility rule" needs receipt and chrome evidence, and a message the BUILD
    never saw has none in the build's pin. So this performs ONE ADDITIONAL READ-ONLY SIDECAR READ
    over the ids its own walk returned and evaluates the real predicate against that. It is a
    read; the zero-write guarantee is untouched.
    """
    from message_processor.channel_stream import (_dedup, _freeze_sidecars, _normalize_page,
                                                  _web, classify_chrome)

    method = _web(client, "conversations_replies")
    raw = await page_messages(method, channel_id=channel_id,
                              extra_params={"ts": str(origin_root_ts)},
                              latest=h, inclusive=True,
                              budget=FetchBudget(deadline_at=deadline_at, page_ceiling=None),
                              label="probe origin walk")
    # THE SAME normalization the build uses, including the "0" floor — an independent walk that
    # normalized differently would report a mismatch that says more about this file than about
    # the build.
    messages = _dedup(_normalize_page(client, raw, channel_id=channel_id, team_id=team_id,
                                      origin=ORIGIN_REPLIES, floor_ts="0",
                                      floor_inclusive=True, high=h))
    payload = await db.read_channel_sidecars_for_async(
        team_id, channel_id, sorted({m.ts for m in messages}))
    pin = _freeze_sidecars(payload)
    # THE BUILD'S FROZEN MARKERS, passed in — never a fresh `serializer_config_snapshot()`. A
    # marker-list change between the build and this walk would otherwise classify the same
    # message two ways and report a membership mismatch that says nothing about the build.
    chrome_ts = classify_chrome(messages, chrome_markers=chrome_markers)
    eligible = [m.ts for m in messages
                if eligible_for_stream(m, pin, receipt_feature_epoch_ts=pin.receipt_feature_epoch_ts,
                                       chrome_ts=chrome_ts)]
    return sorted(eligible), _ts_sha256(eligible)


async def _build_side_membership(shared: Any, origin_fetch: Any, db: Any) -> Tuple[List[str], str]:
    """Re-evaluate the BUILD's OWN origin eligibility over the already-fetched snapshot, against
    a FRESH read-only sidecar read.

    NO NEW SLACK FETCH: the question is whether the SAME messages, evaluated against CURRENT
    evidence, still yield the same eligible set. That is what distinguishes "the world moved
    after the build pinned" from "the build was wrong", and re-running only the independent side
    cannot tell them apart — the change that matters happens between the BUILD's pin and the
    first independent read, a window a walk-1-versus-walk-2 comparison never looks at.
    """
    from message_processor.channel_stream import _freeze_sidecars, classify_chrome

    messages = list(origin_fetch.messages)
    payload = await db.read_channel_sidecars_for_async(
        shared.team_id, shared.channel_id, sorted({m.ts for m in messages}))
    pin = _freeze_sidecars(payload)
    chrome_ts = classify_chrome(messages,
                                chrome_markers=shared.serializer_config["chrome_markers"])
    eligible = [m.ts for m in messages
                if eligible_for_stream(m, pin, receipt_feature_epoch_ts=pin.receipt_feature_epoch_ts,
                                       chrome_ts=chrome_ts)]
    return sorted(eligible), _ts_sha256(eligible)


async def _origin_verdict(client: Any, db: Any, shared: Any, origin_fetch: Any, stream: Any, *,
                          h: str, deadline_at: float,
                          errors: List[str]) -> Dict[str, Any]:
    """`origin_complete` and `origin_stability`, per §4.11's three-row table.

    ONLY A MISMATCH THAT SURVIVES A BUILD-SIDE RE-PIN IS A BUILD FAILURE. The probe observes the
    world at a later instant than the build did, so any difference it cannot reproduce against
    fresh evidence is a difference in the WORLD, not a defect in the build — and blaming the
    build for a receipt that finalized after its pin would be the probe lying in the expensive
    direction. Everything else is `unstable`, which acceptance treats as INCONCLUSIVE (re-run),
    never as a failure.
    """
    root = origin_fetch.origin_root_ts
    built = sorted(item.metadata.get("ts") for item in stream.origin_items
                   if (item.metadata or {}).get("ts"))
    built_hash = _ts_sha256(built)
    out: Dict[str, Any] = {"origin_count": stream.origin_count,
                           "origin_sha256": built_hash}
    if not root:
        out.update({"expected_origin_count": 0, "expected_origin_sha256": _ts_sha256(()),
                    "origin_complete": True, "origin_stability": "stable"})
        return out
    try:
        walk_1, walk_1_hash = await _independent_origin_walk(
            client, db, team_id=shared.team_id, channel_id=shared.channel_id,
            origin_root_ts=root, h=h, deadline_at=deadline_at,
            chrome_markers=shared.serializer_config["chrome_markers"])
    except Exception as e:  # noqa: BLE001
        errors.append(f"independent origin walk failed for {root}: {e}")
        out.update({"expected_origin_count": None, "expected_origin_sha256": None,
                    "origin_complete": None, "origin_stability": "unstable"})
        return out

    out.update({"expected_origin_count": len(walk_1), "expected_origin_sha256": walk_1_hash})
    if walk_1_hash == built_hash:
        out.update({"origin_complete": True, "origin_stability": "stable"})
        return out

    # A MISMATCH. Re-pin the build side, and re-walk once, before blaming anyone.
    try:
        build_2, build_2_hash = await _build_side_membership(shared, origin_fetch, db)
        walk_2, walk_2_hash = await _independent_origin_walk(
            client, db, team_id=shared.team_id, channel_id=shared.channel_id,
            origin_root_ts=root, h=h, deadline_at=deadline_at,
            chrome_markers=shared.serializer_config["chrome_markers"])
    except Exception as e:  # noqa: BLE001
        errors.append(f"origin re-pin failed for {root}: {e}")
        out.update({"origin_complete": None, "origin_stability": "unstable"})
        return out

    if build_2_hash == walk_1_hash:
        # The world moved AFTER the build's pin, and the build was honest AT its pin.
        out.update({"origin_complete": None, "origin_stability": "unstable"})
    elif walk_2_hash != walk_1_hash:
        # Slack changed underneath the probe.
        out.update({"origin_complete": None, "origin_stability": "unstable"})
    elif build_2_hash == built_hash:
        # The mismatch SURVIVES the build-side re-pin. This one is the build's.
        out.update({"origin_complete": False, "origin_stability": "stable"})
    else:
        out.update({"origin_complete": None, "origin_stability": "unstable"})
    return out


async def run_probe(*, client: Any, db: Any, team_id: str, channel_id: str,
                    origins: Sequence[Optional[str]]) -> Dict[str, Any]:
    """One shared periphery, N origin pins. Returns the report payload."""
    errors: List[str] = []

    # H IS A WALL-CLOCK UPPER BOUND, and the field name says so. `conversations.history(limit=1)`
    # looks more precise and is unsafe: it returns the newest HISTORY-VISIBLE message, and Slack's
    # history API does not surface an ordinary reply under an older root. On a channel whose most
    # recent activity is exactly that, an H taken that way sits BELOW the newest real event and
    # silently excludes it — a clean-looking report omitting the very messages the probe exists to
    # exercise. A wall-clock bound is `>=` every event's ts by construction; its only cost is
    # sitting slightly ahead of the newest real message, which for a diagnostic is harmless.
    h = f"{time.time():.6f}"

    anchor_payload = await db.read_channel_window_anchor_async(team_id, channel_id)
    anchor = anchor_payload.get("anchor") or None
    inventory_row = anchor_payload.get("inventory") or None
    from message_processor.channel_stream import InventoryPin, _checked_ts

    coverage = None
    if inventory_row and inventory_row.get("inventory_start_ts"):
        coverage = InventoryPin(
            start_ts=_checked_ts(inventory_row["inventory_start_ts"], "inventory_start_ts"),
            status=str(inventory_row.get("bootstrap_status") or ""),
            reason=inventory_row.get("reason"))
    floor_read = None
    if anchor and int(anchor.get("selection_version") or 0) == SELECTION_VERSION:
        floor_read = str(anchor.get("floor_ts") or "") or None

    # CONSTRUCTED, NOT AWAITED: there is no watermark to drain and no barrier to signal in a
    # fresh CLI process, so the prepare phase is assembled rather than run.
    prepared = PreparedTurn(
        team_id=team_id, channel_id=channel_id, h=h, frontier=0, floor_read=floor_read,
        coverage=coverage, generation=actor_tail_module.generation(channel_id),
        selection_version=SELECTION_VERSION,
        serializer_config=serializer_config_snapshot())

    deadline_at = time.monotonic() + float(config.fetch_retry_total_seconds)
    shared = await build_channel_pin(prepared, client=client, db=db, probe=True,
                                     deadline_at=deadline_at,
                                     reach_tools=reach_tools_for())

    entries: List[Dict[str, Any]] = []
    stream_hashes = set()
    for origin_root_ts in origins:
        # Each origin gets its OWN budget carrying the SAME deadline — `build_origin_pin` raises
        # if they disagree, which is a wiring assertion rather than a timeout.
        # `trigger_ts=None` is deliberate: a diagnostic replay has no trigger, so §2e's
        # empty-origin fallback (legal only when origin_root_ts == trigger_ts) is unreachable,
        # and an empty origin here is the failure it would be for any reply-triggered turn.
        origin_fetch = await fetch_origin_thread(
            client, channel_id, origin_root_ts, h,
            FetchBudget(deadline_at=deadline_at, page_ceiling=None), None)
        pin, origin_pages = await build_origin_pin(shared, origin_fetch, db=db, client=client)
        stream = serialize_stream(pin)
        stream_hashes.add(stream.stream_sha256)
        entry = {"origin_root_ts": origin_root_ts, "origin_pages": origin_pages,
                 "origin_byte_count": stream.origin_byte_count,
                 "union_sha256": stream.union_sha256}
        entry.update(await _origin_verdict(client, db, shared, origin_fetch, stream,
                                           h=h, deadline_at=deadline_at, errors=errors))
        entries.append((entry, stream))

    first_stream = entries[0][1]
    report: Dict[str, Any] = {
        "probe_version": PROBE_VERSION,
        "channel_id": channel_id,
        "h": h, "h_source": "wall_clock_upper_bound", "frontier": 0,
        "periphery_floor_ts": shared.periphery_floor_ts,
        "selection_version": shared.selection_version,
        "root_count": shared.root_count,
        "message_count": first_stream.message_count,
        "candidate_count": first_stream.candidate_count,
        "orphan_root_count": first_stream.orphan_root_count,
        "inventory_state": first_stream.inventory_state,
        "history_pages": shared.pages.history, "reply_pages": shared.pages.reply,
        "byte_count": first_stream.byte_count,
        "stream_sha256": first_stream.stream_sha256,
        "reselected": shared.reselected,
        # ALWAYS FALSE BY CONSTRUCTION: the step that could set it belongs to a function this
        # never invokes. It is reported rather than omitted so the report's shape matches the
        # telemetry event's — and so a `true` here would be an unmistakable signal that the
        # probe wrote something.
        "anchor_advanced": False,
        "errors": errors,
    }

    if len(entries) == 1:
        entry, stream = entries[0]
        report.update({k: v for k, v in entry.items() if k != "origin_pages"})
        report["origin_pages"] = entry["origin_pages"]
        report["origin_byte_count"] = stream.origin_byte_count
    else:
        # In two-origin mode the per-origin fields have no single value and are OMITTED from the
        # top level; `origin_pages` carries the SUM, so the top-level page block still means
        # "pages this command spent". Every PERIPHERY field stays at top level, because there is
        # exactly one periphery.
        report["origins"] = [entry for entry, _stream in entries]
        report["origin_pages"] = sum(entry["origin_pages"] for entry, _ in entries)
        report["prefix_identical"] = len(stream_hashes) == 1
        report["unions_differ"] = len({entry["union_sha256"] for entry, _ in entries}) == len(
            entries)
    return report


async def _amain(args: argparse.Namespace) -> int:
    from slack_client.base import SlackBot

    # The real client, read-only in practice: nothing below calls a write method, and the socket
    # handler is never started, so no event handler it registers can fire.
    client = SlackBot()
    await client._ensure_self_identity()
    team_id = getattr(client, "self_team_id", None) or ""
    db = DatabaseManager(platform="slack")

    report = await run_probe(client=client, db=db, team_id=team_id,
                             channel_id=args.channel, origins=list(args.origin) or [None])

    out = Path(args.out) if args.out else Path(
        f"data/stream_probe_{args.channel}_{int(time.time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nreport written to {out}")
    return 0 if not report["errors"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--channel", required=True, help="channel id to replay")
    parser.add_argument("--origin", action="append", default=[],
                        help="origin thread ts; give TWICE for render-equality mode")
    parser.add_argument("--out", default=None, help="report path")
    args = parser.parse_args(argv)
    if len(args.origin) > 2:
        parser.error("--origin may be given at most twice")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
