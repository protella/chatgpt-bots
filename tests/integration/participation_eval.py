"""Live eval harness for the participation gate.

RETIRED PENDING A RE-BASELINE, and deliberately left un-rewritten. It scores the RICH gate: it
replays scenarios through `classify_participation` and grades the returned
respond/react/react_and_respond/ignore/backoff action against each scenario's `must_be` set. The
binary gate answers a different question — one bit, "does the responder run" — so none of those
labels exist any more and the harness cannot run at all (the function it imports is gone).

It is NOT mechanically portable, which is why this is a note rather than a patch. Mapping the old
labels onto wake/no-wake is easy (anything that spoke or reacted → wake, `ignore` → no wake) but
it would grade the new gate against the old gate's standard, and the standard is exactly what the
binary design changes: the gate is now meant to be GENEROUS, because a false wake costs one call
and can end in the responder's own declared silence, while a false sleep loses the answer. Several
scenarios whose `must_be` is `{"ignore"}` are now legitimately wakes. Re-baselining that corpus is
a judgment call for whoever owns the quality bar, not a rename.

The corpus itself (`participation_scenarios.py`) is still good data and still pure: the
2026-07-25 #ai-tooling incident messages in it are the whole reason this gate was
rebuilt.

Neither this file nor the corpus is part of `make test`; nothing imports them at runtime.

    python3 tests/integration/participation_eval.py                    # 5 trials, live config
    python3 tests/integration/participation_eval.py -n 8 -e medium     # override effort
    python3 tests/integration/participation_eval.py --only addressee   # one category
    python3 tests/integration/participation_eval.py --baseline out.json --compare new.json

Two error classes are reported separately because they are not equally expensive. A FALSE
POSITIVE (spoke when it should have been silent) is what put the bot in front of a channel that
didn't want it. A FALSE NEGATIVE (silent when it should have answered) is a missed answer someone
can always ask for again. A change that trades several false negatives for one false positive is
usually still a win; the reverse rarely is.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config as config_mod  # noqa: E402
from config import config  # noqa: E402
from message_processor.participation import ParticipationEngine  # noqa: E402
from openai_client import OpenAIClient  # noqa: E402
from tests.integration.participation_scenarios import SCENARIOS  # noqa: E402

SPEAKING = {"respond", "react_and_respond"}
# Every way of entering a conversation, emoji included. A scenario whose `must_be` is exactly
# {"ignore"} is saying an emoji would ALSO be an intrusion, so a react there is a false
# positive — unwanted participation — not a mere "wrong kind". Scoring it as the latter let
# reactions into other people's exchanges without ever moving the FP count.
PARTICIPATING = SPEAKING | {"react"}


def _classify(scn, action):
    """correct | false_positive | false_negative | wrong_kind"""
    if action in scn["must_be"]:
        return "correct"
    wanted_speech = bool(scn["must_be"] & SPEAKING)
    silence_only = scn["must_be"] == {"ignore"}
    if silence_only and action in PARTICIPATING:
        return "false_positive"
    if action in SPEAKING and not wanted_speech:
        return "false_positive"
    if action not in SPEAKING and wanted_speech:
        return "false_negative"
    return "wrong_kind"


async def run(trials: int, effort: str | None, only: str | None, concurrency: int):
    if effort:
        config_mod.config.participation_reasoning_effort = effort
    scns = [s for s in SCENARIOS if not only or s["category"] == only or s["id"] == only]
    client = OpenAIClient()
    sem = asyncio.Semaphore(concurrency)

    async def one(scn):
        async with sem:
            try:
                raw = await client.classify_participation(scn["text"], dict(scn["signals"]))
                # Run the RAW dict through the same validate_verdict the gate uses, so what is
                # scored is the verdict the bot would actually act on. Measuring
                # classify_participation alone silently graded the prompt only and gave the
                # invariant layer no credit or blame — a scenario whose declared fields the
                # invariants would have caught still counted as a failure.
                v = ParticipationEngine.validate_verdict(raw)
                # The staged findings, not just the action: an invariant can only refuse a verdict
                # that contradicts itself, so a case that fails with self-consistent fields is a
                # PROMPT problem, and one that fails with contradictory fields is a plumbing bug.
                # Those need opposite fixes, and the action alone cannot tell them apart.
                stages = "/".join(str(getattr(v, k, None) or "-") for k in
                                  ("relation", "exchange_state", "answerability"))
                overruled = f" OVERRULED({','.join(v.overruled_by)})" if v.overruled_by else ""
                return v.action, f"[{stages}]{overruled} {v.reason}"
            except Exception as e:                       # noqa: BLE001
                return f"ERROR:{type(e).__name__}", str(e)[:120]

    started = time.time()
    results = await asyncio.gather(*[one(s) for s in scns for _ in range(trials)])
    elapsed = time.time() - started

    print(f"model={config.utility_model}  effort={config.participation_reasoning_effort}  "
          f"trials={trials}  scenarios={len(scns)}  ({elapsed:.0f}s)\n")

    per_scn, tallies, cats = {}, collections.Counter(), collections.defaultdict(collections.Counter)
    i = 0
    for scn in scns:
        outcomes, actions, sample = collections.Counter(), collections.Counter(), {}
        for _ in range(trials):
            action, reason = results[i]
            i += 1
            o = _classify(scn, action)
            outcomes[o] += 1
            actions[action] += 1
            tallies[o] += 1
            cats[scn["category"]][o] += 1
            sample.setdefault(o, (action, reason))
        per_scn[scn["id"]] = {"outcomes": dict(outcomes), "actions": dict(actions)}
        ok = outcomes["correct"]
        mark = "ok  " if ok == trials else ("FAIL" if ok == 0 else "flak")
        real = " *" if scn.get("real") else "  "
        print(f"  [{mark}]{real}{scn['id']:<32} {ok}/{trials}  {dict(actions)}")
        for o in ("false_positive", "false_negative", "wrong_kind"):
            if outcomes[o]:
                a, r = sample[o]
                print(f"          └─ {o}: {a} — {r[:96]}")

    total = sum(tallies.values())
    print(f"\n  {'TOTAL':<34} {tallies['correct']}/{total} "
          f"({100*tallies['correct']/max(total,1):.0f}%)")
    print(f"  {'false positives (spoke, should not)':<34} {tallies['false_positive']}")
    print(f"  {'false negatives (silent, should speak)':<34} {tallies['false_negative']}")
    if tallies["wrong_kind"]:
        print(f"  {'wrong kind (missed react/backoff)':<34} {tallies['wrong_kind']}")
    print("\n  by category:")
    for cat in sorted(cats):
        c = cats[cat]
        n = sum(c.values())
        print(f"    {cat:<16} {c['correct']}/{n}   fp={c['false_positive']} "
              f"fn={c['false_negative']} wk={c['wrong_kind']}")

    return {"config": {"model": config.utility_model,
                       "effort": config.participation_reasoning_effort, "trials": trials},
            "totals": dict(tallies), "scenarios": per_scn}


def _compare(base_path, new_path):
    base, new = (json.load(open(p)) for p in (base_path, new_path))
    print(f"{'scenario':<34} {'base':>9}  {'new':>9}   delta")
    for sid, nv in new["scenarios"].items():
        bv = base["scenarios"].get(sid)
        if not bv:
            continue
        b = bv["outcomes"].get("correct", 0)
        n = nv["outcomes"].get("correct", 0)
        bt = sum(bv["outcomes"].values())
        nt = sum(nv["outcomes"].values())
        d = (n / max(nt, 1)) - (b / max(bt, 1))
        flag = "  <<< REGRESSION" if d < -0.01 else ("  +" if d > 0.01 else "")
        print(f"{sid:<34} {b:>4}/{bt:<4} {n:>4}/{nt:<4}  {d:+.0%}{flag}")
    for k in ("correct", "false_positive", "false_negative"):
        print(f"{k:<34} {base['totals'].get(k,0):>9}  {new['totals'].get(k,0):>9}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--trials", type=int, default=5)
    ap.add_argument("-e", "--effort", default=None)
    ap.add_argument("--only", default=None, help="category or scenario id")
    ap.add_argument("-c", "--concurrency", type=int, default=12)
    ap.add_argument("-o", "--out", default=None, help="write JSON results here")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    if a.baseline and a.compare:
        _compare(a.baseline, a.compare)
        sys.exit(0)
    res = asyncio.run(run(a.trials, a.effort, a.only, a.concurrency))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
