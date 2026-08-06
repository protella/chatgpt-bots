"""Operator tooling: diagnostics and repo guards, imported as `tools.<name>`.

The package marker is load-bearing for mypy — without it `tools/pii_scan.py` resolves both as
top-level `pii_scan` and as `tools.pii_scan` (the name its test imports), which mypy rejects.
"""
