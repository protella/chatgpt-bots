"""The live battery (SHALLOW_STREAM_RESPEC §7, §9).

Code in this package TALKS TO SLACK. Nothing here runs under `make test` — the capped unit gate
is `pytest tests/unit`, and the harness's network-free tests live at
`tests/unit/test_battery_harness.py` deliberately, so they cannot rot outside the gate.
"""
