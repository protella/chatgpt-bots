"""The gate that runs before every armed battery pass — does the LIVE bot see us as a human?

    python3 -m tests.live.classify_probe

`DEV_TREAT_BOT_IDS_AS_HUMAN` is read once at the bot's import, from the environment it booted
with, so the battery's preflight can only prove the `.env` FILE names the right ids. **This proves
the running process loaded them.** It posts one ordinary throwaway remark as the operator and
reads that message's own `gate_start` row out of the participation ledger: `sender_type` must be
`human`. Anything else means every row of the pass would grade the bot's behaviour toward a bot —
silently, which is exactly what voided the 2026-08-01 pass.

**IT COSTS A CLASSIFICATION, NOT A TURN.** The remark is addressed to nobody and worth nothing, so
the gate declines it and the bot posts no reply.

TWO RULES IT OBEYS, the same two the battery obeys (owner, 2026-08-02): the message reads as a
person talking — no marker token, nothing about tests in it — and it is **not deleted afterwards**.
Its ts is printed so the run record can name it.

It lives in `tests/live/` rather than in a session scratchpad because it is part of the harness
contract: the run book tells the operator to run it before every armed pass, and a run book that
points at an ephemeral file points at nothing.
"""
from __future__ import annotations

import asyncio
import sys
import time
from typing import Any, Dict, Optional, Sequence

from tests.live.battery_harness import (REPLY_POLL_SECONDS, HarnessError, pace, post_seed,
                                        read_ledger_events)

# The one channel live testing is authorized in — the same rail `run_battery` carries, for the
# same reason: this posts a message, and a typo must not post it into a real conversation.
DEFAULT_CHANNEL = "C0BKX77NU66"

# How long to wait for the bot to judge the probe. It is a gate decision, not a turn, so this is
# generous rather than tuned: a bot that has not judged an ordinary message within two minutes is
# not a bot the pass should run against either.
VERDICT_DEADLINE_SECONDS = 120.0

# Ordinary throwaway remarks addressed to nobody. Varied so a run does not repeat the last run's
# words verbatim in a channel where nothing is ever deleted.
ASIDES = (
    "coffee machine is making that noise again",
    "whoever restocked the kitchen, thank you",
    "rain absolutely came out of nowhere",
    "lift on the third floor is doing its slow thing again",
    "someone left a very nice umbrella by the door",
)


def aside_for(now: float) -> str:
    return ASIDES[int(now) % len(ASIDES)]


def gate_row(channel: str, trigger_ts: str) -> Optional[Dict[str, Any]]:
    """This message's own `gate_start`, or None while the bot has not judged it yet.

    KEYED ON `trigger_ts`, NOT `ts`. The ledger names the message that STARTED the attempt; a
    reader keying on `ts` matches nothing and waits out its deadline with the row sitting there
    the whole time.
    """
    for event in read_ledger_events("gate_start"):
        if (event.get("channel_id") == channel
                and str(event.get("trigger_ts")) == str(trigger_ts)):
            return event
    return None


async def classify_probe(channel: str = DEFAULT_CHANNEL,
                         deadline: float = VERDICT_DEADLINE_SECONDS) -> Dict[str, Any]:
    """Post the aside, wait for its gate row, and report what the live bot made of the sender.

    Returns `{"ts", "sender_type", "sender_is_bot", "human"}`. A message the bot never judged
    raises: "we cannot tell" is not the same answer as "it saw a bot", and a pass must not start
    on either.
    """
    ts = await post_seed(channel, aside_for(time.time()))
    bound = time.monotonic() + max(0.0, deadline)
    while True:
        row = gate_row(channel, ts)
        if row is not None:
            sender = str(row.get("sender_type") or "")
            return {"ts": ts, "sender_type": sender,
                    "sender_is_bot": row.get("sender_is_bot"), "human": sender == "human"}
        if time.monotonic() >= bound:
            raise HarnessError(
                f"the bot never judged probe {ts} in {channel} within {deadline}s — no gate_start "
                f"row at all, so whether it would classify the operator as human is unknown")
        await pace(REPLY_POLL_SECONDS)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """0 = human (a pass may run), 1 = classified as something else, 2 = never judged."""
    args = list(argv if argv is not None else sys.argv[1:])
    channel = args[0] if args else DEFAULT_CHANNEL
    if channel != DEFAULT_CHANNEL:
        raise SystemExit(f"refusing to post in {channel!r}: live testing is authorized only in "
                         f"{DEFAULT_CHANNEL} (#chatgpt-bot-test). Prod is hands-off.")
    try:
        result = asyncio.run(classify_probe(channel))
    except HarnessError as e:
        print(f"VERDICT: cannot confirm — {e}")
        return 2
    print(f"probe posted ts={result['ts']} (left in the channel — nothing here deletes)")
    print(f"gate_start sender_type = {result['sender_type']!r}  "
          f"(sender_is_bot={result['sender_is_bot']})")
    if result["human"]:
        print("VERDICT: PASS — the live process classifies a user-token seed as human.")
        return 0
    print(f"VERDICT: FAIL — seeds classify as {result['sender_type']!r}. Do NOT run a pass; the "
          f"allowlist has not taken effect in the running process.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
