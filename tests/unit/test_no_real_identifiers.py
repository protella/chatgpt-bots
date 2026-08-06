"""The PII guard, run as part of every suite.

This repo is PUBLIC and has leaked real coworkers' names into its published history twice. The
scanner in `tools/pii_scan.py` is the fix; this file is what makes it unskippable, because the
one thing both leaks had in common is that nobody ran a check.

A failure here is never "update the test". It is: take the real string out, and use the fictional
cast in `tests/fixtures/people.py`.
"""
import pytest

from tests.fixtures.people import ALLOWED_SPEAKERS, EXTRAS, FIRST_NAMES, NON_HUMAN, ROSTER
from tools import pii_scan


@pytest.mark.critical
def test_no_real_identifiers_in_tracked_files():
    findings = pii_scan.scan()
    assert not findings, (
        "This repository is PUBLIC — real identifiers must never be committed.\n\n"
        + "\n".join(f"  {f}" for f in findings)
        + "\n\nUse the fictional cast in tests/fixtures/people.py."
    )


def test_scanner_actually_scans_something():
    """A guard that silently scans zero files passes forever. Assert it has work to do."""
    files = pii_scan.tracked_files()
    assert len(files) > 100
    assert any(f.startswith("tests/") for f in files)
    assert any(f.endswith("prompts.py") for f in files)


def test_denylist_is_never_published():
    """The denylist holds the real strings. Committing it would publish exactly what it guards."""
    assert "Docs/internal/pii_denylist.txt" not in pii_scan.tracked_files()


def test_roster_is_self_consistent():
    assert ALLOWED_SPEAKERS == frozenset(ROSTER + EXTRAS + NON_HUMAN)
    assert not set(ROSTER) & set(EXTRAS), "a character belongs to one list or the other"
    for name in FIRST_NAMES:
        assert any(n.split()[0] == name for n in ROSTER + EXTRAS)


@pytest.mark.parametrize("line,should_flag", [
    ('    Say("1780000000.000100", "Jane Doe", "hi"),', True),
    ('    Say("1780000000.000100", "Riley Reyes", "hi"),', False),
    ('    Say("1780000000.000100", "ChatGPT", "hi"),', False),
    ('    _src("hello", who="Some Person", ts="1.0"),', True),
    ('    _src("hello", who="Tessa Tran", ts="1.0"),', False),
    ('    PEOPLE = {"U-x": "Real Name"}', True),
    ('    addressees=("Riley Reyes", "Nobody Here")', True),
    ('    name="OPS-7 rollout"', False),
])
def test_roster_rule_catches_new_names(tmp_path, monkeypatch, line, should_flag):
    """The rule that catches a name nobody has flagged yet — the novelty case."""
    target = tmp_path / "tests" / "unit" / "test_sample.py"
    target.parent.mkdir(parents=True)
    target.write_text(line + "\n", encoding="utf-8")
    monkeypatch.setattr(pii_scan, "REPO", tmp_path)
    findings = pii_scan.scan(["tests/unit/test_sample.py"])
    flagged = [f for f in findings if f.rule == "roster"]
    assert bool(flagged) is should_flag, f"{line!r} → {findings}"


# The token-shaped sample is ASSEMBLED rather than written out: a literal one here is indexed by
# GitHub's push protection, which blocked this very commit the first time. A guard that cannot be
# pushed guards nothing.
_TOKEN_SAMPLE = "xox" + "b-" + "Z" * 20


@pytest.mark.parametrize("line,should_flag", [
    (f'token = "{_TOKEN_SAMPLE}"', True),
    ('monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-not-a-real-token")', False),
    ('contact = "someone@acmecorp.com"', True),
    ('contact = "someone@example.com"', False),
])
def test_credential_rule(tmp_path, monkeypatch, line, should_flag):
    target = tmp_path / "sample.py"
    target.write_text(line + "\n", encoding="utf-8")
    monkeypatch.setattr(pii_scan, "REPO", tmp_path)
    findings = pii_scan.scan(["sample.py"])
    flagged = [f for f in findings if f.rule == "credential"]
    assert bool(flagged) is should_flag, f"{line!r} → {findings}"


def test_denylist_rule_catches_a_known_real_string(tmp_path, monkeypatch):
    """The recurrence case: a name an implementer copied out of stale context."""
    (tmp_path / "Docs" / "internal").mkdir(parents=True)
    (tmp_path / "Docs" / "internal" / "pii_denylist.txt").write_text(
        "# comment\nRealPerson\n\n", encoding="utf-8")
    (tmp_path / "sample.md").write_text("as realperson said last week\n", encoding="utf-8")
    monkeypatch.setattr(pii_scan, "REPO", tmp_path)
    monkeypatch.setattr(pii_scan, "DENYLIST_PATH",
                        tmp_path / "Docs" / "internal" / "pii_denylist.txt")
    findings = pii_scan.scan(["sample.md"])
    assert [f.rule for f in findings] == ["denylist"]


def test_scan_runs_without_a_denylist(tmp_path, monkeypatch):
    """A fresh clone has no denylist. Rules 2 and 3 must still run rather than the guard dying."""
    (tmp_path / "sample.py").write_text('x = "harmless"\n', encoding="utf-8")
    monkeypatch.setattr(pii_scan, "REPO", tmp_path)
    monkeypatch.setattr(pii_scan, "DENYLIST_PATH", tmp_path / "nope.txt")
    assert pii_scan.load_denylist() == []
    assert pii_scan.scan(["sample.py"]) == []
