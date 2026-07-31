"""Unit tests for the dev-only turn barriers (spec §13 live checks)."""
import asyncio
import json

import pytest

from message_processor import dev_barriers
from message_processor.dev_barriers import (POST_ADMISSION, POST_PARTIAL_POST,
                                           PRE_RESUME_AFTER_COMPACTION, SEAMS)

_SEAM_FNS = {
    POST_ADMISSION: dev_barriers.post_admission,
    POST_PARTIAL_POST: dev_barriers.post_partial_post,
    PRE_RESUME_AFTER_COMPACTION: dev_barriers.pre_resume_after_compaction,
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("DEV_TURN_BARRIERS", "DEV_TURN_BARRIERS_DIR", "DEV_TURN_BARRIERS_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def barrier_dir(tmp_path):
    return tmp_path / "barriers"


async def _await(coro, timeout=10):
    return await asyncio.wait_for(coro, timeout=timeout)


def test_all_three_seams_are_declared():
    assert SEAMS == (POST_ADMISSION, POST_PARTIAL_POST, PRE_RESUME_AFTER_COMPACTION)
    assert set(_SEAM_FNS) == set(SEAMS)


@pytest.mark.parametrize("seam", SEAMS)
async def test_unset_env_is_a_hard_no_op(seam, monkeypatch, barrier_dir):
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", str(barrier_dir))
    assert await _await(_SEAM_FNS[seam](channel="C1", thread="100.000100")) is False
    assert not barrier_dir.exists()


@pytest.mark.parametrize("seam", SEAMS)
async def test_empty_env_value_is_a_hard_no_op(seam, monkeypatch, barrier_dir):
    monkeypatch.setenv("DEV_TURN_BARRIERS", "  ")
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", str(barrier_dir))
    assert await _await(_SEAM_FNS[seam]()) is False
    assert not barrier_dir.exists()


async def test_enabled_without_a_dir_does_not_pause(monkeypatch):
    monkeypatch.setenv("DEV_TURN_BARRIERS", "all")
    assert await _await(dev_barriers.post_admission(channel="C1")) is False


async def test_blank_dir_does_not_pause(monkeypatch):
    monkeypatch.setenv("DEV_TURN_BARRIERS", POST_ADMISSION)
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", "   ")
    assert await _await(dev_barriers.post_admission()) is False


@pytest.mark.parametrize("value", [POST_ADMISSION, "1", "all", "true",
                                   f" {POST_ADMISSION.upper()} ",
                                   f"other_seam,{POST_ADMISSION}"])
async def test_enabling_values(value, monkeypatch, barrier_dir):
    monkeypatch.setenv("DEV_TURN_BARRIERS", value)
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", str(barrier_dir))
    monkeypatch.setenv("DEV_TURN_BARRIERS_TIMEOUT", "0")
    assert await _await(dev_barriers.post_admission()) is True


@pytest.mark.parametrize("seam", SEAMS)
async def test_each_seam_fires_only_on_its_own_name(seam, monkeypatch, barrier_dir):
    monkeypatch.setenv("DEV_TURN_BARRIERS", seam)
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", str(barrier_dir))
    monkeypatch.setenv("DEV_TURN_BARRIERS_TIMEOUT", "0")
    for name, fn in _SEAM_FNS.items():
        assert await _await(fn()) is (name == seam)


async def test_release_file_ends_the_pause(monkeypatch, barrier_dir):
    monkeypatch.setenv("DEV_TURN_BARRIERS", POST_ADMISSION)
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", str(barrier_dir))
    monkeypatch.setenv("DEV_TURN_BARRIERS_TIMEOUT", "60")
    # §4a: the announcement is KEYED `(seam, operation_id, test_epoch_id)`. Derived rather than
    # spelled out, so a change to the key SHAPE moves this test with it instead of pinning one
    # filename. The RELEASE stays unkeyed on purpose — `<seam>.release` frees every operation at
    # that seam, which is what a harness touches when it does not care which turn it caught.
    key = dev_barriers.barrier_key(POST_ADMISSION, {"channel": "C1"})
    waiting = barrier_dir / f"{POST_ADMISSION}.{key}.waiting"
    release = barrier_dir / f"{POST_ADMISSION}.release"

    task = asyncio.create_task(
        dev_barriers.post_admission(channel="C1", thread_ts="100.000100"))
    for _ in range(200):
        if waiting.exists():
            break
        await asyncio.sleep(0.02)
    assert waiting.exists()
    assert not task.done()

    context = json.loads(waiting.read_text(encoding="utf-8"))
    assert context["seam"] == POST_ADMISSION
    assert context["channel"] == "C1"
    assert context["thread_ts"] == "100.000100"
    assert isinstance(context["at"], float)

    release.write_text("go", encoding="utf-8")
    assert await asyncio.wait_for(task, timeout=30) is True
    assert not waiting.exists()
    # The UNKEYED release SURVIVES, deliberately (§4a). It is a broadcast that frees every
    # operation at this seam, so consuming it here would swallow the signal a second concurrent
    # turn is still waiting on — the collision keyed barriers exist to remove. Only the KEYED
    # release is cleaned up, by the one operation it belongs to.
    assert release.exists()
    assert not (barrier_dir / f"{POST_ADMISSION}.{key}.release").exists()


async def test_non_serializable_context_still_announces(monkeypatch, barrier_dir):
    monkeypatch.setenv("DEV_TURN_BARRIERS", POST_PARTIAL_POST)
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", str(barrier_dir))
    monkeypatch.setenv("DEV_TURN_BARRIERS_TIMEOUT", "60")
    key = dev_barriers.barrier_key(POST_PARTIAL_POST, {})
    waiting = barrier_dir / f"{POST_PARTIAL_POST}.{key}.waiting"
    release = barrier_dir / f"{POST_PARTIAL_POST}.release"

    task = asyncio.create_task(dev_barriers.post_partial_post(ledger=object()))
    for _ in range(200):
        if waiting.exists():
            break
        await asyncio.sleep(0.02)
    assert json.loads(waiting.read_text(encoding="utf-8"))["seam"] == POST_PARTIAL_POST

    release.write_text("go", encoding="utf-8")
    assert await asyncio.wait_for(task, timeout=30) is True


@pytest.mark.parametrize("timeout", ["0", "0.05"])
async def test_timeout_returns_and_cleans_up(timeout, monkeypatch, barrier_dir):
    monkeypatch.setenv("DEV_TURN_BARRIERS", "all")
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", str(barrier_dir))
    monkeypatch.setenv("DEV_TURN_BARRIERS_TIMEOUT", timeout)
    assert await _await(dev_barriers.post_admission(channel="C1")) is True
    assert barrier_dir.exists()
    assert list(barrier_dir.iterdir()) == []


async def test_unknown_seam_never_fires(monkeypatch, barrier_dir):
    monkeypatch.setenv("DEV_TURN_BARRIERS", "all")
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", str(barrier_dir))
    monkeypatch.setenv("DEV_TURN_BARRIERS_TIMEOUT", "0")
    assert await _await(dev_barriers.barrier("not_a_seam", {"channel": "C1"})) is False
    assert not barrier_dir.exists()


async def test_unwritable_dir_does_not_pause(monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("DEV_TURN_BARRIERS", "all")
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", str(blocker / "sub"))
    monkeypatch.setenv("DEV_TURN_BARRIERS_TIMEOUT", "0")
    assert await _await(dev_barriers.post_admission()) is False
