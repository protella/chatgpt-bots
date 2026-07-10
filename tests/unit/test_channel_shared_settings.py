"""Shared per-channel response settings (model / reasoning effort / verbosity).

Anyone in a channel can set them via the channel settings modal; they apply to
everyone there. Hierarchy: env defaults < user prefs < channel shared < thread
overrides, with the composed effort clamped against the composed model.
"""
import sqlite3
from unittest.mock import patch

import pytest

from config import BotConfig, SUPPORTED_CHAT_MODELS
from database import DatabaseManager


@pytest.fixture
def db(tmp_path):
    with patch("os.makedirs"):
        d = DatabaseManager("test")
        d.db_path = str(tmp_path / "test.db")
        if getattr(d, "conn", None):
            d.conn.close()
        d.conn = sqlite3.connect(d.db_path, check_same_thread=False, isolation_level=None)
        d.conn.row_factory = sqlite3.Row
        d.conn.execute("PRAGMA journal_mode=WAL")
        d.init_schema()
        yield d
        if getattr(d, "conn", None):
            d.conn.close()


# ---------------- DB roundtrip ----------------

def test_channel_settings_store_and_clear_shared_overrides(db):
    db.set_channel_settings("C1", model="gpt-5.5", reasoning_effort="high",
                            verbosity="low", updated_by="U1")
    cs = db.get_channel_settings("C1")
    assert (cs["model"], cs["reasoning_effort"], cs["verbosity"]) == ("gpt-5.5", "high", "low")

    # Omitted fields are preserved; explicit None clears to NULL (inherit)
    db.set_channel_settings("C1", model=None)
    cs = db.get_channel_settings("C1")
    assert cs["model"] is None
    assert cs["reasoning_effort"] == "high"  # untouched


def test_channel_settings_shared_fields_default_null(db):
    db.set_channel_settings("C2", participation_level="judicious")
    cs = db.get_channel_settings("C2")
    assert cs["model"] is None and cs["reasoning_effort"] is None and cs["verbosity"] is None


@pytest.mark.asyncio
async def test_channel_settings_async_roundtrip(db):
    await db.set_channel_settings_async("C3", model="gpt-5.6-terra", reasoning_effort="max")
    cs = await db.get_channel_settings_async("C3")
    assert cs["model"] == "gpt-5.6-terra" and cs["reasoning_effort"] == "max"


# ---------------- hierarchy ----------------

@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("GPT_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("DEFAULT_REASONING_EFFORT", "medium")
    return BotConfig()


def test_channel_settings_beat_user_prefs(cfg):
    user_prefs = {"model": "gpt-5.6-luna", "reasoning_effort": "low", "verbosity": "low"}
    channel = {"model": "gpt-5.6-terra", "reasoning_effort": "high", "verbosity": None}
    composed = cfg._compose_thread_config(user_prefs, None, channel)
    assert composed["model"] == "gpt-5.6-terra"
    assert composed["reasoning_effort"] == "high"
    assert composed["verbosity"] == "low"  # NULL channel column → user pref survives


def test_thread_overrides_beat_channel_settings(cfg):
    channel = {"model": "gpt-5.6-terra", "reasoning_effort": "high"}
    overrides = {"model": "gpt-5.5"}
    composed = cfg._compose_thread_config(None, overrides, channel)
    assert composed["model"] == "gpt-5.5"


def test_cross_layer_effort_is_clamped_to_model(cfg):
    # Channel pins gpt-5.5 while a user's personal effort is `max` (5.6-only):
    # the composed config must clamp, never 400.
    user_prefs = {"reasoning_effort": "max"}
    channel = {"model": "gpt-5.5"}
    composed = cfg._compose_thread_config(user_prefs, None, channel)
    assert composed["model"] == "gpt-5.5"
    assert composed["reasoning_effort"] == "xhigh"


def test_no_channel_settings_is_a_noop(cfg):
    baseline = cfg._compose_thread_config({"model": "gpt-5.6-luna"}, None, None)
    assert baseline["model"] == "gpt-5.6-luna"
    assert cfg._map_channel_settings(None) == {}
    # Participation-only rows contribute nothing to generation config
    assert cfg._map_channel_settings({"participation_level": "active", "model": None}) == {}


def test_get_thread_config_fetches_channel_row(cfg, db):
    db.set_channel_settings("C9", model="gpt-5.6-terra")
    composed = cfg.get_thread_config(user_id=None, db=db, channel_id="C9")
    assert composed["model"] == "gpt-5.6-terra"
    # And a DM/unknown channel falls back to defaults
    composed = cfg.get_thread_config(user_id=None, db=db, channel_id="D_NOPE")
    assert composed["model"] == "gpt-5.6-sol"


# ---------------- modal ----------------

def test_channel_modal_offers_shared_selects_and_personal_button():
    from settings_modal import SettingsModal
    modal = SettingsModal.__new__(SettingsModal)  # builder needs no db
    view = modal.build_channel_settings_modal(
        "C1", {"model": "gpt-5.5", "reasoning_effort": "high", "verbosity": None},
        "tag_only",
    )
    by_block = {b.get("block_id"): b for b in view["blocks"] if b.get("block_id")}

    model_el = by_block["channel_model_block"]["element"]
    assert model_el["initial_option"]["value"] == "gpt-5.5"
    assert {o["value"] for o in model_el["options"]} == {"inherit", *SUPPORTED_CHAT_MODELS}

    assert by_block["channel_effort_block"]["element"]["initial_option"]["value"] == "high"
    # NULL verbosity renders as inherit
    assert by_block["channel_verbosity_block"]["element"]["initial_option"]["value"] == "inherit"

    actions = by_block["personal_settings_link_block"]
    assert actions["elements"][0]["action_id"] == "open_user_settings_push"
