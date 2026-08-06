#!/usr/bin/env python3
"""Fail the build when a real person, channel or credential reaches this PUBLIC repository.

WHY THIS EXISTS. Twice now, real coworkers' names have been committed here — in scenario tests
reconstructed from real Slack incidents, and once inside a shipped system prompt. The second time
was weeks after a full history rewrite had cleaned up the first. The leak is not carelessness
about a rule; it is that the rule lived only in a human's head while the writing was done by
whoever happened to be implementing that round. So it is a test now, and it runs in every suite.

THREE RULES, DELIBERATELY DIFFERENT IN KIND:

  1. DENYLIST — exact known-real strings (names, internal channels, workspace ids) read from
     `Docs/internal/pii_denylist.txt`, which is gitignored and therefore never published. This
     catches RECURRENCE: the observed failure is an implementer seeing a real name in old context
     and reusing it. The file is optional — a fresh clone has no copy, and rules 2 and 3 still
     run — because a guard that cannot run without a secret file is a guard that gets deleted.

  2. ROSTER — any person-shaped name in a speaker field under tests/ must come from
     `tests/fixtures/people.py`. This catches NOVELTY: a name nobody has flagged yet, typed while
     writing a scenario. The check is scoped to speaker fields (`who=`, `Say(...)`, roster maps,
     `real_name`, ...) rather than to all prose, because "Deep Research" and "Slack Connect" are
     capitalized pairs too and a check that cries wolf is a check that gets skipped.

  3. PATTERNS — email addresses and API credentials, anywhere. These have no legitimate form in
     this repo, so they are matched structurally rather than from a list.

Run directly (`python3 -m tools.pii_scan`) for a report, or let
`tests/unit/test_no_real_identifiers.py` run it as part of the suite.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, NamedTuple

REPO = Path(__file__).resolve().parent.parent
DENYLIST_PATH = REPO / "Docs" / "internal" / "pii_denylist.txt"

# Rule 2's scope. A speaker field is a place where the value IS a person, so a capitalized pair
# found here is a name and not a product. Each pattern captures the name in group 1.
SPEAKER_PATTERNS = (
    re.compile(r'(?:who|username|user_name|real_name|display_name|author|sender)\s*=\s*"([^"]+)"'),
    re.compile(r'"(?:real_name|display_name|user_name|username)"\s*:\s*"([^"]+)"'),
    re.compile(r'Say\(\s*"[0-9.]+"\s*,\s*"([^"]+)"'),
    re.compile(r'"U[-_][A-Za-z0-9]+"\s*:\s*"([^"]+)"'),
    re.compile(r'addressees\s*=\s*\(([^)]*)\)'),
)

# A value that looks like a human name: two or more capitalized words. "OPS-7", "C_INSIGHTS" and
# "the deploy channel" are not, and are left alone.
PERSON_SHAPED = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][A-Za-z'’-]+)+$")

CREDENTIAL_PATTERNS = (
    ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{12,}")),
    ("OpenAI key", re.compile(r"\bsk-[A-Za-z0-9_-]{24,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    # A stub domain is one whose second-level label is three characters or fewer — "a@x.com",
    # "erin@b.com". Nobody's real employer is called that, and tests are full of them, so the
    # length test separates the fixture from the leak without a domain allowlist to maintain.
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@(?!localhost)"
                                 r"(?:[A-Za-z0-9-]+\.)*[A-Za-z0-9-]{4,}\.[A-Za-z]{2,}\b")),
)

# Credentials that are obviously inert. A test needs to pass SOMETHING token-shaped.
CREDENTIAL_ALLOW = re.compile(r"not-a-real|fake|dummy|placeholder|your-|xxx|1234|example|"
                              r"@test\.|@localhost", re.I)

# Binary and generated files carry no reviewable text.
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".ico", ".db", ".xml"}
SKIP_PREFIXES = ("htmlcov/", "data/", "logs/", ".venv/")

# The guard's own files necessarily contain the vocabulary they police: the scanner spells out
# the patterns, the roster lists the cast, and the test asserts on deliberately-bad sample lines.
SELF = {"tools/pii_scan.py", "tests/fixtures/people.py", "Docs/internal/pii_denylist.txt",
        "tests/unit/test_no_real_identifiers.py"}


class Finding(NamedTuple):
    path: str
    line: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.detail}"


def tracked_files() -> List[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    files = []
    for name in out.splitlines():
        if name in SELF or name.startswith(SKIP_PREFIXES):
            continue
        if Path(name).suffix.lower() in SKIP_SUFFIXES:
            continue
        files.append(name)
    return files


def load_denylist() -> List[str]:
    """Known-real strings, one per line; `#` comments and blanks ignored. Absent file → no rule 1."""
    if not DENYLIST_PATH.exists():
        return []
    terms = []
    for raw in DENYLIST_PATH.read_text(encoding="utf-8").splitlines():
        term = raw.split("#", 1)[0].strip()
        if term:
            terms.append(term)
    return terms


def allowed_speakers() -> frozenset:
    sys.path.insert(0, str(REPO))
    from tests.fixtures.people import ALLOWED_SPEAKERS  # noqa: E402
    return ALLOWED_SPEAKERS


def _speaker_values(line: str) -> Iterable[str]:
    for pattern in SPEAKER_PATTERNS:
        for match in pattern.finditer(line):
            raw = match.group(1)
            # addressees=(...) holds a tuple of quoted names; the rest hold one value.
            for value in re.findall(r'"([^"]+)"', raw) or [raw]:
                yield value


def scan(paths: Iterable[str] | None = None) -> List[Finding]:
    files = list(paths) if paths is not None else tracked_files()
    denylist = load_denylist()
    roster = allowed_speakers()
    denials = [(term, re.compile(re.escape(term), re.I)) for term in denylist]
    findings: List[Finding] = []

    for name in files:
        path = REPO / name
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        in_tests = name.startswith("tests/")

        for lineno, line in enumerate(text.splitlines(), 1):
            for term, pattern in denials:
                if pattern.search(line):
                    findings.append(Finding(name, lineno, "denylist",
                                            f"real identifier {term!r} — see Docs/internal/"
                                            f"pii_denylist.txt"))
            for label, pattern in CREDENTIAL_PATTERNS:
                match = pattern.search(line)
                if match and not CREDENTIAL_ALLOW.search(line):
                    findings.append(Finding(name, lineno, "credential",
                                            f"{label}: {match.group(0)!r}"))
            if in_tests:
                for value in _speaker_values(line):
                    if PERSON_SHAPED.match(value) and value not in roster:
                        findings.append(Finding(
                            name, lineno, "roster",
                            f"speaker {value!r} is not in tests/fixtures/people.py — pick an "
                            f"existing character, or add one there deliberately"))
    return findings


def main() -> int:
    findings = scan()
    if not findings:
        n = len(tracked_files())
        deny = len(load_denylist())
        print(f"pii_scan: clean — {n} tracked files, {deny} denylist terms")
        return 0
    print(f"pii_scan: {len(findings)} problem(s)\n", file=sys.stderr)
    for finding in findings:
        print(f"  {finding}", file=sys.stderr)
    print("\nThis repository is PUBLIC. Nothing above may be committed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
