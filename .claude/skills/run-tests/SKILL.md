---
name: run-tests
description: Run this repo's pytest suite safely. Use for ANY pytest invocation here — an uncapped run once grew to 30GB and OOM-killed the machine. Covers the memory cap, orphan cleanup, and the mock-stream failure mode behind it.
---

# Running tests in chatgpt-bots-dev

Every `pytest` invocation gets a virtual-memory cap. No exceptions, including single-test runs.

```bash
(ulimit -v 4194304; timeout 600 python3 -m pytest tests/unit -q -p no:cacheprovider --tb=no -p no:logging)
```

`make test` (coverage) and `make test-fast` are fine to reach for, but wrap them the same way.

## Why the cap

An uncapped `python3 -m pytest tests/unit` reached **30 GB RSS** and the kernel OOM-killed it,
taking WSL down with it — repeatedly. The cap costs nothing and converts a runaway into a
`MemoryError` traceback that names the culprit test instead of a dead machine.

`--tb=no` is **not** sufficient on its own: assertion rewriting builds the failure explanation
inside the test, before any reporting happens. The `ulimit` is the real guard.

## Orphan discipline

Interrupting the agent turn does **not** kill the process it spawned. A stopped turn left an
orphaned pytest eating the box for another 400 seconds.

- `pgrep -af pytest` before and after any run.
- Kill orphans explicitly, and run `pkill` **alone** — inside a compound command it kills the
  calling shell (exit 144).
- Never run the suite in the background, and never leave one running.

## The failure mode it guards against

A mock stream that never terminates. `output_text += <MagicMock>` does not raise — it silently
turns `output_text` into a mock, and a stale `side_effect` once left an async iterator with no end.

Any test mocking a response stream must terminate and must yield real strings. More in
`Docs/TOOL_SUBSYSTEMS.md`.
