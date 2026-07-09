"""Model lineup cleanup — gpt-5.5 is the only selectable chat model; gpt-5-mini is
utility-only. Covers the supported-model surface (picker, validation, token limits)
and the startup normalization migration for stale user/thread model selections."""
import json
import sqlite3
import tempfile
from unittest.mock import MagicMock

import pytest

from config import BotConfig, MODEL_KNOWLEDGE_CUTOFFS
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
    assert set(MODEL_KNOWLEDGE_CUTOFFS) == {"gpt-5.5", "gpt-5-mini", "default"}


def test_model_picker_offers_only_gpt55(modal):
    blocks = modal._build_modal_blocks(
        settings={"model": "gpt-5.5"}, selected_model="gpt-5.5",
        is_new_user=False, in_thread=False, scope="global",
    )
    model_block = next(b for b in blocks if b.get("block_id") == "model_block")
    options = [o["value"] for o in model_block["accessory"]["options"]]
    assert options == ["gpt-5.5"]


def test_modal_coerces_stale_model_in_blocks(modal):
    """A stale thread override (dropped model) must not produce an invalid
    initial_option — the picker only contains gpt-5.5."""
    blocks = modal._build_modal_blocks(
        settings={"model": "gpt-5.5"}, selected_model="gpt-5.5",
        is_new_user=False, in_thread=True, scope="thread",
    )
    model_block = next(b for b in blocks if b.get("block_id") == "model_block")
    initial = model_block["accessory"]["initial_option"]["value"]
    assert initial == "gpt-5.5"


def test_validate_settings_strips_temp_when_reasoning_active(modal):
    validated = modal.validate_settings(
        {"model": "gpt-5.5", "reasoning_effort": "low", "temperature": 0.5, "top_p": 0.9}
    )
    assert "temperature" not in validated
    assert "top_p" not in validated


def test_validate_settings_keeps_temp_with_reasoning_none(modal):
    validated = modal.validate_settings(
        {"model": "gpt-5.5", "reasoning_effort": "none", "temperature": 0.5, "top_p": 0.9}
    )
    assert validated["temperature"] == 0.5
    assert validated["top_p"] == 0.9


def test_token_limits_gpt55_and_utility():
    config = BotConfig()
    assert config.get_model_token_limit("gpt-5.5") == int(
        config.gpt54_max_tokens * config.gpt54_token_buffer_percentage
    )
    mini = config.get_model_token_limit("gpt-5-mini")
    assert mini == int(config.gpt5_max_tokens * config.token_buffer_percentage)
    # Unknown models fall back to the conservative utility window
    assert config.get_model_token_limit("some-future-model") == mini


# --- startup normalization migration ---

def _insert_user(db, user_id, model):
    db.conn.execute(
        "INSERT INTO user_preferences (slack_user_id, model) VALUES (?, ?)",
        (user_id, model),
    )


def test_migration_normalizes_dropped_user_models(temp_db):
    _insert_user(temp_db, "U1", "gpt-4o")
    _insert_user(temp_db, "U2", "gpt-5.4")
    _insert_user(temp_db, "U3", "gpt-5.5")
    temp_db._run_migrations()

    rows = {
        r["slack_user_id"]: r["model"]
        for r in temp_db.conn.execute("SELECT slack_user_id, model FROM user_preferences")
    }
    assert rows == {"U1": "gpt-5.5", "U2": "gpt-5.5", "U3": "gpt-5.5"}


def test_migration_normalizes_thread_overrides(temp_db):
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C1:1", "C1", "1", json.dumps({"model": "gpt-5.1", "temperature": 0.7})),
    )
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C1:2", "C1", "2", json.dumps({"reasoning_effort": "low"})),  # no model key
    )
    temp_db._run_migrations()

    row = temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C1:1'"
    ).fetchone()
    cfg = json.loads(row["config_json"])
    assert cfg["model"] == "gpt-5.5"
    assert cfg["temperature"] == 0.7  # other keys untouched

    row2 = temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C1:2'"
    ).fetchone()
    assert "model" not in json.loads(row2["config_json"])  # untouched


def test_migration_is_idempotent(temp_db):
    _insert_user(temp_db, "U1", "gpt-4.1")
    temp_db._run_migrations()
    temp_db._run_migrations()  # second run must be a no-op, not an error
    row = temp_db.conn.execute(
        "SELECT model FROM user_preferences WHERE slack_user_id = 'U1'"
    ).fetchone()
    assert row["model"] == "gpt-5.5"
