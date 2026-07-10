"""Model lineup — the GPT-5.6 family (sol/terra/luna) plus gpt-5.5 are the selectable
chat models; gpt-5.6-luna doubles as the utility model. Covers the supported-model
surface (picker, validation, token limits) and the startup migrations/normalizers for
stale user/thread model selections (one-time everyone->sol swap + every-startup clamp)."""
import json
import sqlite3
import tempfile
from unittest.mock import MagicMock

import pytest

from config import BotConfig, MODEL_KNOWLEDGE_CUTOFFS, SUPPORTED_CHAT_MODELS
from database import DatabaseManager
from settings_modal import SettingsModal


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager("test")
        db.db_path = f"{tmpdir}/test.db"
        db.conn = sqlite3.connect(
            db.db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        db.conn.row_factory = sqlite3.Row
        db.init_schema()
        yield db
        db.conn.close()


@pytest.fixture
def modal():
    return SettingsModal(db=MagicMock())


# --- supported-model surface ---

def test_knowledge_cutoffs_only_supported_models():
    assert set(MODEL_KNOWLEDGE_CUTOFFS) == {
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "default"
    }


def test_model_picker_offers_supported_lineup(modal):
    blocks = modal._build_modal_blocks(
        settings={"model": "gpt-5.6-sol"}, selected_model="gpt-5.6-sol",
        is_new_user=False, in_thread=False, scope="global",
    )
    model_block = next(b for b in blocks if b.get("block_id") == "model_block")
    options = [o["value"] for o in model_block["accessory"]["options"]]
    assert options == SUPPORTED_CHAT_MODELS
    assert options == ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"]


def test_modal_keeps_gpt55_selectable(modal):
    """gpt-5.5 stays a valid initial_option (still supported)."""
    blocks = modal._build_modal_blocks(
        settings={"model": "gpt-5.5"}, selected_model="gpt-5.5",
        is_new_user=False, in_thread=True, scope="thread",
    )
    model_block = next(b for b in blocks if b.get("block_id") == "model_block")
    initial = model_block["accessory"]["initial_option"]["value"]
    assert initial == "gpt-5.5"


def test_validate_settings_strips_temp_when_reasoning_active(modal):
    validated = modal.validate_settings(
        {"model": "gpt-5.6-sol", "reasoning_effort": "low", "temperature": 0.5, "top_p": 0.9}
    )
    assert "temperature" not in validated
    assert "top_p" not in validated


def test_validate_settings_keeps_temp_with_reasoning_none(modal):
    # Verified live 2026-07-09: 5.6 accepts temperature/top_p at effort=none
    for model in ("gpt-5.6-sol", "gpt-5.5"):
        validated = modal.validate_settings(
            {"model": model, "reasoning_effort": "none", "temperature": 0.5, "top_p": 0.9}
        )
        assert validated["temperature"] == 0.5
        assert validated["top_p"] == 0.9


def test_token_limits_families_and_fallback():
    config = BotConfig()
    big = int(config.gpt54_max_tokens * config.gpt54_token_buffer_percentage)
    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"):
        assert config.get_model_token_limit(model) == big
    # Unknown models fall back to the conservative window
    small = int(config.gpt5_max_tokens * config.token_buffer_percentage)
    assert config.get_model_token_limit("some-future-model") == small


# --- startup migrations ---

def _insert_user(db, user_id, model, effort=None):
    if effort is None:
        db.conn.execute(
            "INSERT INTO user_preferences (slack_user_id, model) VALUES (?, ?)",
            (user_id, model),
        )
    else:
        db.conn.execute(
            "INSERT INTO user_preferences (slack_user_id, model, reasoning_effort) VALUES (?, ?, ?)",
            (user_id, model, effort),
        )


def test_one_time_migration_swaps_everyone_to_sol_medium(temp_db):
    """The sentinel-gated migration moves ALL users to gpt-5.6-sol + medium
    reasoning — including users already on a still-supported model."""
    # Simulate a pre-upgrade database: init_schema planted the sentinel on this
    # fresh DB, so remove it to exercise the one-time path real DBs will take.
    temp_db.conn.execute("ALTER TABLE user_preferences DROP COLUMN gpt56_migrated")
    _insert_user(temp_db, "U1", "gpt-4o", "high")
    _insert_user(temp_db, "U2", "gpt-5.5", "xhigh")
    temp_db._run_migrations()

    rows = {
        r["slack_user_id"]: (r["model"], r["reasoning_effort"])
        for r in temp_db.conn.execute(
            "SELECT slack_user_id, model, reasoning_effort FROM user_preferences")
    }
    assert rows == {
        "U1": ("gpt-5.6-sol", "medium"),
        "U2": ("gpt-5.6-sol", "medium"),
    }


def test_one_time_migration_runs_once(temp_db):
    """After the sentinel exists, users who re-pick gpt-5.5 keep it."""
    temp_db._run_migrations()  # plants gpt56_migrated
    _insert_user(temp_db, "U1", "gpt-5.5", "xhigh")
    temp_db._run_migrations()  # must NOT re-swap
    row = temp_db.conn.execute(
        "SELECT model, reasoning_effort FROM user_preferences WHERE slack_user_id='U1'"
    ).fetchone()
    assert (row["model"], row["reasoning_effort"]) == ("gpt-5.5", "xhigh")


def test_normalizer_coerces_dropped_models_after_sentinel(temp_db):
    temp_db._run_migrations()
    _insert_user(temp_db, "U1", "gpt-5.1")     # dropped -> sol
    _insert_user(temp_db, "U2", "gpt-5.6-terra")  # supported -> untouched
    temp_db._run_migrations()
    rows = {
        r["slack_user_id"]: r["model"]
        for r in temp_db.conn.execute("SELECT slack_user_id, model FROM user_preferences")
    }
    assert rows == {"U1": "gpt-5.6-sol", "U2": "gpt-5.6-terra"}


def test_normalizer_clamps_stored_efforts(temp_db):
    temp_db._run_migrations()
    _insert_user(temp_db, "U1", "gpt-5.6-luna", "minimal")  # 400 on 5.6 -> none
    _insert_user(temp_db, "U2", "gpt-5.5", "max")           # no max on 5.5 -> xhigh
    temp_db._run_migrations()
    rows = {
        r["slack_user_id"]: r["reasoning_effort"]
        for r in temp_db.conn.execute(
            "SELECT slack_user_id, reasoning_effort FROM user_preferences")
    }
    assert rows == {"U1": "none", "U2": "xhigh"}


def test_migration_normalizes_thread_overrides(temp_db):
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C1:1", "C1", "1", json.dumps({"model": "gpt-5.1", "temperature": 0.7})),
    )
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C1:2", "C1", "2", json.dumps({"reasoning_effort": "low"})),  # no model key
    )
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C1:3", "C1", "3", json.dumps({"model": "gpt-5.5", "reasoning_effort": "max"})),
    )
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C1:4", "C1", "4", json.dumps({"model": "gpt-5.6-luna", "reasoning_effort": "minimal"})),
    )
    temp_db._run_migrations()

    cfg = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C1:1'").fetchone()["config_json"])
    assert cfg["model"] == "gpt-5.6-sol"
    assert cfg["temperature"] == 0.7  # other keys untouched

    cfg2 = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C1:2'").fetchone()["config_json"])
    assert "model" not in cfg2  # untouched

    cfg3 = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C1:3'").fetchone()["config_json"])
    assert cfg3 == {"model": "gpt-5.5", "reasoning_effort": "xhigh"}  # kept 5.5, clamped max

    cfg4 = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C1:4'").fetchone()["config_json"])
    assert cfg4 == {"model": "gpt-5.6-luna", "reasoning_effort": "none"}  # clamped minimal


def test_migration_is_idempotent(temp_db):
    _insert_user(temp_db, "U1", "gpt-4.1")
    temp_db._run_migrations()
    temp_db._run_migrations()  # second run must be a no-op, not an error
    row = temp_db.conn.execute(
        "SELECT model, reasoning_effort FROM user_preferences WHERE slack_user_id = 'U1'"
    ).fetchone()
    assert (row["model"], row["reasoning_effort"]) == ("gpt-5.6-sol", "medium")
