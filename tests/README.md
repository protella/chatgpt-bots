# Tests

**Every pytest invocation gets a virtual-memory cap — no exceptions, including single-test runs.**
An uncapped run once reached 30GB RSS and the kernel OOM-killed the machine. The cap costs nothing
and turns a runaway into a `MemoryError` that names the culprit test. The Make recipes do **not**
cap themselves, so wrap those too:

```bash
(ulimit -v 4194304; timeout 600 python3 -m pytest tests/unit -q -p no:cacheprovider --tb=no -p no:logging)
(ulimit -v 4194304; make test)       # unit + coverage
(ulimit -v 4194304; make test-all)   # adds integration; needs real keys in .env
```

`make test-fast` and `make test-all` pass no path, so `testpaths = tests` sends them across the
whole tree (integration included; the live battery is separate — see `tests/live/README.md`). For the quick loop, name `tests/unit` yourself:
`(ulimit -v 4194304; python3 -m pytest tests/unit -q)`.

- Markers (`pytest.ini`): `unit`, `integration`, `slow`, `asyncio`, `skip_ci`, `critical`,
  `smoke` — e.g. `(ulimit -v 4194304; python3 -m pytest tests/unit -m critical -q)`.
- Full rationale, orphan discipline, and the mock-stream failure mode behind the cap: the
  `run-tests` skill.
- Live dev-bot battery against real Slack: `tests/live/README.md`.
- Every test speaker comes from the fictional cast in `tests/fixtures/people.py` — this repo is
  public, so never write a real name, channel, or credential into a test.
- Review contract for changes under `tests/`: `AGENTS.md`.
