"""Model lineup — gpt-6-astra, the GPT-5.6 family (sol/terra/luna) and gpt-5.5 are the
selectable chat models; gpt-5.6-luna doubles as the utility model. Covers the supported-model
surface (picker, validation, token limits) and the startup migrations/normalizers for
stale user/thread model selections (one-time everyone->sol swap + every-startup clamp).

The every-startup normalizer reads its allowlist and its reset target from config at RUN TIME.
A hard-coded list there silently resets every user off a newly added model on the next restart
— which is exactly what would have happened to an Astra selection — so these tests compare
against the constants rather than against a literal lineup.
"""
import json
import sqlite3
import tempfile
from unittest.mock import MagicMock

import pytest

from config import BotConfig, MODEL_KNOWLEDGE_CUTOFFS, SUPPORTED_CHAT_MODELS
from database import DatabaseManager
from slack_client.settings_modal import SettingsModal


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
        "gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "default"
    }


def test_model_picker_offers_supported_lineup(modal):
    blocks = modal._build_modal_blocks(
        settings={"model": "gpt-5.6-sol"}, selected_model="gpt-5.6-sol",
        is_new_user=False, in_thread=False, scope="global",
    )
    model_block = next(b for b in blocks if b.get("block_id") == "model_block")
    options = [o["value"] for o in model_block["accessory"]["options"]]
    assert options == SUPPORTED_CHAT_MODELS
    assert options == ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
                       "gpt-5.5"]


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

def _columns(db, table):
    return [col[1] for col in db.conn.execute(f"PRAGMA table_info({table})")]


#: `image_model` left out of the INSERT entirely, so the column DEFAULT applies — which is
#: not the same row state as an explicit NULL, and the migration has to cover both.
_COLUMN_DEFAULT = object()


def _insert_user(db, user_id, model, effort=None, image_model=_COLUMN_DEFAULT,
                 image_quality=_COLUMN_DEFAULT, image_size=_COLUMN_DEFAULT,
                 image_background=_COLUMN_DEFAULT):
    columns = ["slack_user_id", "model"]
    values = [user_id, model]
    if effort is not None:
        columns.append("reasoning_effort")
        values.append(effort)
    if image_model is not _COLUMN_DEFAULT:
        columns.append("image_model")
        values.append(image_model)
    if image_quality is not _COLUMN_DEFAULT:
        columns.append("image_quality")
        values.append(image_quality)
    if image_size is not _COLUMN_DEFAULT:
        columns.append("image_size")
        values.append(image_size)
    if image_background is not _COLUMN_DEFAULT:
        columns.append("image_background")
        values.append(image_background)
    placeholders = ", ".join("?" * len(columns))
    db.conn.execute(
        f"INSERT INTO user_preferences ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )


def test_one_time_sol_reset_stands_down_once_astra_has_migrated(temp_db):
    """The historical everyone->sol/medium reset is SUPERSEDED by the Astra swap.

    Its sentinel is missing here but `gpt6_migrated` is present, so the Astra migration
    has already carried each user's chosen effort forward. Re-running the sol reset on
    top would flatten those rows to sol/medium and throw the carry-over away, so it only
    plants its sentinel. What still moves is the every-startup normalizer, which coerces
    a dropped model to `config.gpt_model`."""
    # Simulate a pre-upgrade database: init_schema planted the sentinel on this
    # fresh DB, so remove it to exercise the one-time path real DBs will take.
    temp_db.conn.execute("ALTER TABLE user_preferences DROP COLUMN gpt56_migrated")
    _insert_user(temp_db, "U1", "gpt-4o", "high")
    _insert_user(temp_db, "U2", "gpt-5.5", "xhigh")
    temp_db._run_migrations()

    from config import config as bot_config
    rows = {
        r["slack_user_id"]: (r["model"], r["reasoning_effort"])
        for r in temp_db.conn.execute(
            "SELECT slack_user_id, model, reasoning_effort FROM user_preferences")
    }
    assert rows == {
        "U1": (bot_config.gpt_model, "high"),   # dropped model coerced, effort KEPT
        "U2": ("gpt-5.5", "xhigh"),             # supported model left alone entirely
    }
    assert 'gpt56_migrated' in _columns(temp_db, "user_preferences")


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
    # The reset target is `config.gpt_model`, read at run time — not a literal.
    from config import config as bot_config
    assert rows == {"U1": bot_config.gpt_model, "U2": "gpt-5.6-terra"}


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

    from config import config as bot_config
    cfg = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C1:1'").fetchone()["config_json"])
    assert cfg["model"] == bot_config.gpt_model   # dropped model -> the configured default
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
    from config import config as bot_config
    row = temp_db.conn.execute(
        "SELECT model, reasoning_effort FROM user_preferences WHERE slack_user_id = 'U1'"
    ).fetchone()
    assert (row["model"], row["reasoning_effort"]) == (bot_config.gpt_model, "medium")


# --- gpt-6 survives the normalizer (codex C1) ---

def test_the_normalizer_preserves_a_stored_astra_selection(temp_db):
    """The allowlist used to be a hard-coded four-model literal, and `_migrate_gpt56` runs on
    EVERY migration pass. A user could select Astra, use it, restart the bot, and find
    themselves silently back on Sol."""
    temp_db._run_migrations()
    _insert_user(temp_db, "U1", "gpt-6-astra", "high")
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C1:9", "C1", "9", json.dumps({"model": "gpt-6-astra", "reasoning_effort": "xhigh"})),
    )
    temp_db._run_migrations()

    row = temp_db.conn.execute(
        "SELECT model, reasoning_effort FROM user_preferences WHERE slack_user_id = 'U1'"
    ).fetchone()
    assert (row["model"], row["reasoning_effort"]) == ("gpt-6-astra", "high")

    cfg = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C1:9'").fetchone()["config_json"])
    assert cfg == {"model": "gpt-6-astra", "reasoning_effort": "xhigh"}


def test_the_normalizer_clamps_a_stored_none_effort_on_gpt6(temp_db):
    """`none` is a hard 400 on gpt-6 and it is the default UTILITY effort, so it is sitting in
    real rows today. OpenAI's own migration guidance is to start at `low`."""
    temp_db._run_migrations()
    _insert_user(temp_db, "U1", "gpt-6-astra", "none")
    _insert_user(temp_db, "U2", "gpt-6-astra", "minimal")
    _insert_user(temp_db, "U3", "gpt-5.6-sol", "none")     # still legal there — untouched
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C2:1", "C2", "1", json.dumps({"model": "gpt-6-astra", "reasoning_effort": "none"})),
    )
    temp_db._run_migrations()

    rows = {
        r["slack_user_id"]: r["reasoning_effort"]
        for r in temp_db.conn.execute(
            "SELECT slack_user_id, reasoning_effort FROM user_preferences")
    }
    assert rows == {"U1": "low", "U2": "low", "U3": "none"}

    cfg = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C2:1'").fetchone()["config_json"])
    assert cfg == {"model": "gpt-6-astra", "reasoning_effort": "low"}


# --- one-time GPT-6 Astra migration (everyone -> astra, effort preserved) ---

def _drop_gpt6_sentinel(db):
    """Simulate a pre-Astra database. `init_schema` already ran the migrations on this
    fresh DB, so the sentinel is planted — remove it to exercise the one-time path that
    real databases will take on their first boot after the upgrade."""
    db.conn.execute("ALTER TABLE user_preferences DROP COLUMN gpt6_migrated")


def test_gpt6_migration_moves_every_user_to_astra_keeping_their_effort(temp_db):
    """Everyone lands on Astra AND on Sunburst. The effort each user chose survives unless
    Astra rejects it — `none` and `minimal` are not on Astra's ladder and step up to
    `low`. The image model has no such nuance: every row is set, a NULL one included, so
    the choice comes out explicit rather than falling back to a config default.

    Image quality goes to `high` for EVERYONE (owner ruling, 2026-09-09), whatever the row
    held: the old shipped default sat in almost every row and was never a pick."""
    _drop_gpt6_sentinel(temp_db)
    _insert_user(temp_db, "U1", "gpt-5.6-sol", "medium",
                 image_model="gpt-image-2", image_quality="medium")
    _insert_user(temp_db, "U2", "gpt-5.5", "xhigh", image_model=None, image_quality=None)
    _insert_user(temp_db, "U3", "gpt-5.6-luna", "none")
    _insert_user(temp_db, "U4", "gpt-5.6-terra", "minimal",
                 image_model="gpt-image-1", image_quality="low")
    _insert_user(temp_db, "U5", "gpt-5.6-sol", "high", image_quality="auto")
    _insert_user(temp_db, "U6", "gpt-5.6-sol", "high", image_quality="")
    temp_db._run_migrations()

    rows = {
        r["slack_user_id"]: (
            r["model"], r["reasoning_effort"], r["image_model"], r["image_quality"])
        for r in temp_db.conn.execute(
            "SELECT slack_user_id, model, reasoning_effort, image_model, image_quality "
            "FROM user_preferences")
    }
    assert rows == {
        # a previously stored quality moves too — everyone lands on high
        "U1": ("gpt-6-astra", "medium", "gpt-image-2.5-sunburst", "high"),
        "U4": ("gpt-6-astra", "low", "gpt-image-2.5-sunburst", "high"),
        "U2": ("gpt-6-astra", "xhigh", "gpt-image-2.5-sunburst", "high"),
        "U3": ("gpt-6-astra", "low", "gpt-image-2.5-sunburst", "high"),
        "U5": ("gpt-6-astra", "high", "gpt-image-2.5-sunburst", "high"),
        "U6": ("gpt-6-astra", "high", "gpt-image-2.5-sunburst", "high"),
    }


def test_gpt6_migration_moves_saved_image_sizes_onto_the_new_defaults(temp_db):
    """The image half of the swap: shape, tier and background.

    `1024x1024` was the old shipped default rather than anyone's pick, so it joins the
    unset rows on `auto` — the shape is chosen per request from then on. A grid size the
    user really did choose keeps its SHAPE and moves to the Large cell of it. An off-grid
    size is a deliberate number and is left exactly as it is.
    """
    _drop_gpt6_sentinel(temp_db)
    _insert_user(temp_db, "U_OLD_DEFAULT", "gpt-5.6-sol", image_size="1024x1024",
                 image_background="opaque")
    _insert_user(temp_db, "U_UNSET", "gpt-5.6-sol", image_size=None)
    _insert_user(temp_db, "U_AUTO", "gpt-5.6-sol", image_size="auto")
    _insert_user(temp_db, "U_4K", "gpt-5.6-sol", image_size="3840x2160")
    _insert_user(temp_db, "U_PORTRAIT", "gpt-5.6-sol", image_size="1024x1536")
    _insert_user(temp_db, "U_OFFGRID", "gpt-5.6-sol", image_size="1234x5678")
    temp_db._run_migrations()

    rows = {
        r["slack_user_id"]: (r["image_size"], r["image_tier"], r["image_background"])
        for r in temp_db.conn.execute(
            "SELECT slack_user_id, image_size, image_tier, image_background "
            "FROM user_preferences")
    }
    assert rows == {
        "U_OLD_DEFAULT": ("auto", "large", "auto"),
        "U_UNSET": ("auto", "large", "auto"),
        "U_AUTO": ("auto", "large", "auto"),
        "U_4K": ("1920x1088", "large", "auto"),
        "U_PORTRAIT": ("1184x1776", "large", "auto"),
        "U_OFFGRID": ("1234x5678", "large", "auto"),
    }


def test_gpt6_migration_moves_a_thread_size_but_never_pins_a_thread_tier(temp_db):
    """A thread that pins a size gets the same rule the user column does. It must NOT
    gain an `$.image_tier`: a thread which never pinned one inherits the user's tier, and
    writing one here would freeze that inheritance at migration time."""
    _drop_gpt6_sentinel(temp_db)
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C7:1", "C7", "1", json.dumps({"image_size": "3840x2160"})),
    )
    temp_db._run_migrations()

    cfg = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C7:1'").fetchone()["config_json"])
    assert cfg == {"image_size": "1920x1088"}


def test_gpt6_migration_rewrites_a_thread_size_present_as_json_null(temp_db):
    """A PRESENT `$.image_size` whose value is JSON `null` is a pinned-but-empty size, and
    it has to move with NULL/''/auto/1024x1024. `json_extract` returns SQL NULL for it, so
    an extract-based gate would skip it; the gate reads `json_type` instead. A thread
    WITHOUT the key is untouched — it inherits the user/channel size and must keep doing so."""
    _drop_gpt6_sentinel(temp_db)
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C8:1", "C8", "1", json.dumps({"image_size": None, "model": "gpt-5.6-sol"})),
    )
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C8:2", "C8", "2", json.dumps({"model": "gpt-5.6-sol"})),
    )
    temp_db._run_migrations()

    present = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C8:1'").fetchone()["config_json"])
    assert present["image_size"] == "auto"

    absent = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C8:2'").fetchone()["config_json"])
    assert "image_size" not in absent


def test_gpt6_migration_rewrites_thread_overrides_in_place(temp_db):
    """json_set edits the four keys it owns — every other field in the override document
    has to come out the other side unchanged."""
    _drop_gpt6_sentinel(temp_db)
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C3:1", "C3", "1", json.dumps({
            "model": "gpt-5.6-sol",
            "reasoning_effort": "none",
            "image_model": "gpt-image-1",
            "image_quality": "auto",
            "image_size": "1024x1024",
            "verbosity": "high",
            "temperature": 0.7,
        })),
    )
    temp_db._run_migrations()

    cfg = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C3:1'").fetchone()["config_json"])
    assert cfg == {
        "model": "gpt-6-astra",
        "reasoning_effort": "low",
        "image_model": "gpt-image-2.5-sunburst",
        "image_quality": "high",
        # 1024x1024 was the OLD SHIPPED DEFAULT, not a pick — it becomes the auto shape.
        "image_size": "auto",
        "verbosity": "high",
        "temperature": 0.7,
    }


def test_gpt6_migration_moves_every_pinned_thread_quality_to_high(temp_db):
    """Owner ruling, 2026-09-09: `$.image_quality` moves to `high` wherever the key is
    present, whatever it held — a pinned `medium` included. A thread WITHOUT the key keeps
    inheriting and must not gain one."""
    _drop_gpt6_sentinel(temp_db)
    for thread_id, quality in (("C9:1", "medium"), ("C9:2", "auto"), ("C9:3", None)):
        temp_db.conn.execute(
            "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) "
            "VALUES (?, ?, ?, ?)",
            (thread_id, "C9", thread_id.split(":")[1],
             json.dumps({"image_quality": quality})),
        )
    # A thread without the key at all keeps inheriting, so it must not gain one.
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C9:4", "C9", "4", json.dumps({"verbosity": "high"})),
    )
    temp_db._run_migrations()

    qualities = {
        r["thread_id"]: json.loads(r["config_json"]).get("image_quality", "ABSENT")
        for r in temp_db.conn.execute(
            "SELECT thread_id, config_json FROM threads WHERE channel_id = 'C9'")
    }
    assert qualities == {
        "C9:1": "high", "C9:2": "high", "C9:3": "high", "C9:4": "ABSENT"}


def test_gpt6_migration_swaps_channel_overrides_and_leaves_null_models_alone(temp_db):
    """A NULL channel model means 'no channel override' — swapping it would invent one.
    That holds for the image pin on its own column too."""
    _drop_gpt6_sentinel(temp_db)
    temp_db.conn.execute(
        "INSERT INTO channel_settings (channel_id, model, reasoning_effort, image_model) "
        "VALUES (?, ?, ?, ?)",
        ("C3", "gpt-5.6-sol", "high", "gpt-image-2"),
    )
    temp_db.conn.execute(
        "INSERT INTO channel_settings (channel_id, model, reasoning_effort, image_model) "
        "VALUES (?, ?, ?, ?)",
        ("C4", None, None, None),
    )
    temp_db._run_migrations()

    rows = {
        r["channel_id"]: (r["model"], r["reasoning_effort"], r["image_model"])
        for r in temp_db.conn.execute(
            "SELECT channel_id, model, reasoning_effort, image_model FROM channel_settings")
    }
    assert rows == {
        "C3": ("gpt-6-astra", "high", "gpt-image-2.5-sunburst"),
        "C4": (None, None, None),
    }


def test_a_brand_new_user_row_starts_on_sunburst_at_high_quality(temp_db):
    """The migration only moves rows that already exist. A user created afterwards has to
    land on the same place, or the very next signup quietly starts on the old image model.
    Both the column DEFAULT and the explicit INSERT in `create_default_user_preferences`
    have to agree on that — this asserts the row, so either one drifting is caught."""
    from config import config as bot_config
    temp_db._run_migrations()
    temp_db.create_default_user_preferences("U_NEW", "someone@example.com")

    # Compared against config, not against literals: DEFAULT_IMAGE_QUALITY and
    # GPT_IMAGE_MODEL are env-overridable, so a literal here would be asserting the
    # developer's `.env` rather than that the INSERT carries the configured values.
    row = temp_db.conn.execute(
        "SELECT image_model, image_quality FROM user_preferences WHERE slack_user_id = 'U_NEW'"
    ).fetchone()
    assert (row["image_model"], row["image_quality"]) == (
        bot_config.image_model, bot_config.default_image_quality)

    # A row inserted without touching the image columns at all falls to the DEFAULTs,
    # which is the path the explicit INSERT above would mask.
    temp_db.conn.execute(
        "INSERT INTO user_preferences (slack_user_id) VALUES ('U_BARE')")
    bare = temp_db.conn.execute(
        "SELECT image_model, image_quality FROM user_preferences WHERE slack_user_id = 'U_BARE'"
    ).fetchone()
    assert (bare["image_model"], bare["image_quality"]) == ("gpt-image-2.5-sunburst", "high")


def test_gpt6_migration_survives_a_database_without_the_channel_image_column(temp_db):
    """`channel_settings.image_model` is added by a LATER migration step than this swap,
    so on an old enough database the column is not there yet. The image UPDATE has to
    stand down instead of raising — inside the swap's transaction a "no such column"
    takes the whole one-time migration down with it, users included."""
    _drop_gpt6_sentinel(temp_db)
    temp_db.conn.execute("ALTER TABLE channel_settings DROP COLUMN image_model")
    _insert_user(temp_db, "U1", "gpt-5.6-sol", "high", image_model="gpt-image-2")
    temp_db._run_migrations()

    assert 'gpt6_migrated' in _columns(temp_db, "user_preferences")
    row = temp_db.conn.execute(
        "SELECT model, image_model FROM user_preferences WHERE slack_user_id = 'U1'"
    ).fetchone()
    assert (row["model"], row["image_model"]) == ("gpt-6-astra", "gpt-image-2.5-sunburst")


def test_gpt6_migration_does_not_run_a_second_time(temp_db):
    """The sentinel column is the guard. Once it exists, a user who picks another model
    back must keep it across every later restart."""
    _drop_gpt6_sentinel(temp_db)
    _insert_user(temp_db, "U1", "gpt-5.6-sol", "high")
    temp_db._run_migrations()
    assert temp_db.conn.execute(
        "SELECT model FROM user_preferences WHERE slack_user_id = 'U1'"
    ).fetchone()["model"] == "gpt-6-astra"

    temp_db.conn.execute(
        "UPDATE user_preferences SET model = 'gpt-5.6-sol' WHERE slack_user_id = 'U1'")
    temp_db._run_migrations()

    row = temp_db.conn.execute(
        "SELECT model, reasoning_effort FROM user_preferences WHERE slack_user_id = 'U1'"
    ).fetchone()
    assert (row["model"], row["reasoning_effort"]) == ("gpt-5.6-sol", "high")


# --- the Astra swap is atomic (codex: blocker) ---

class _FailOnceConnection:
    """Connection proxy that raises the first time a statement containing `needle` runs.

    The migration's own recovery path (`rollback`, `commit`, `in_transaction`) has to keep
    working, so everything except the poisoned `execute` falls through to the real
    connection untouched.
    """

    def __init__(self, conn, needle):
        self._conn = conn
        self._needle = needle
        self.fired = False

    def execute(self, sql, *args, **kwargs):
        if not self.fired and self._needle in sql:
            self.fired = True
            raise sqlite3.OperationalError("injected migration failure")
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_gpt6_migration_rolls_back_entirely_when_a_later_update_fails(temp_db):
    """The sentinel column must never outlive a half-finished swap.

    The connection is in autocommit and `_migration_step` swallows the exception, so
    without one explicit transaction the ALTER would commit, the thread UPDATE would
    blow up, and every later boot would see the sentinel and skip the users that never
    got migrated. SQLite's DDL is transactional, so the rollback takes the column too.
    """
    _drop_gpt6_sentinel(temp_db)
    _insert_user(temp_db, "U1", "gpt-5.6-sol", "high",
                 image_model="gpt-image-2", image_quality="low")
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C9:1", "C9", "1", json.dumps({"model": "gpt-5.6-sol", "reasoning_effort": "high"})),
    )

    real_conn = temp_db.conn
    temp_db.conn = _FailOnceConnection(real_conn, "UPDATE threads")
    try:
        temp_db._run_migrations()   # _migration_step logs and continues; must not raise
        assert temp_db.conn.fired, "the injected failure never fired"
    finally:
        temp_db.conn = real_conn

    assert 'gpt6_migrated' not in _columns(temp_db, "user_preferences")
    row = temp_db.conn.execute(
        "SELECT model, reasoning_effort, image_model, image_quality FROM user_preferences "
        "WHERE slack_user_id = 'U1'"
    ).fetchone()
    assert (row["model"], row["reasoning_effort"], row["image_model"],
            row["image_quality"]) == ("gpt-5.6-sol", "high", "gpt-image-2", "low")
    cfg = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C9:1'").fetchone()["config_json"])
    assert cfg["model"] == "gpt-5.6-sol"

    # The retry has to find the migration still pending and finish it.
    temp_db._run_migrations()

    assert 'gpt6_migrated' in _columns(temp_db, "user_preferences")
    row = temp_db.conn.execute(
        "SELECT model, reasoning_effort, image_model, image_quality FROM user_preferences "
        "WHERE slack_user_id = 'U1'"
    ).fetchone()
    # Quality moves for everyone, `low` included.
    assert (row["model"], row["reasoning_effort"], row["image_model"],
            row["image_quality"]) == ("gpt-6-astra", "high", "gpt-image-2.5-sunburst", "high")
    cfg = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C9:1'").fetchone()["config_json"])
    assert cfg == {"model": "gpt-6-astra", "reasoning_effort": "high"}


# --- the Astra swap runs BEFORE the 5.6 normalizers (codex: ordering) ---

def test_gpt55_max_survives_the_swap_as_astra_max(temp_db):
    """`max` is not on gpt-5.5's ladder but IS on Astra's. With the 5.6 normalizer going
    first it clamped the stored value to `xhigh` before the swap could read it, and the
    user quietly lost a reasoning level they had chosen."""
    _drop_gpt6_sentinel(temp_db)
    _insert_user(temp_db, "U1", "gpt-5.5", "max")
    temp_db.conn.execute(
        "INSERT INTO threads (thread_id, channel_id, thread_ts, config_json) VALUES (?, ?, ?, ?)",
        ("C8:1", "C8", "1", json.dumps({"model": "gpt-5.5", "reasoning_effort": "max"})),
    )
    temp_db.conn.execute(
        "INSERT INTO channel_settings (channel_id, model, reasoning_effort) VALUES (?, ?, ?)",
        ("C8", "gpt-5.5", "max"),
    )
    temp_db._run_migrations()

    row = temp_db.conn.execute(
        "SELECT model, reasoning_effort FROM user_preferences WHERE slack_user_id = 'U1'"
    ).fetchone()
    assert (row["model"], row["reasoning_effort"]) == ("gpt-6-astra", "max")

    cfg = json.loads(temp_db.conn.execute(
        "SELECT config_json FROM threads WHERE thread_id = 'C8:1'").fetchone()["config_json"])
    assert cfg == {"model": "gpt-6-astra", "reasoning_effort": "max"}

    chan = temp_db.conn.execute(
        "SELECT model, reasoning_effort FROM channel_settings WHERE channel_id = 'C8'").fetchone()
    assert (chan["model"], chan["reasoning_effort"]) == ("gpt-6-astra", "max")


def test_database_predating_both_sentinels_lands_on_astra_with_efforts_kept(temp_db):
    """A DB old enough to lack `gpt56_migrated` used to get the everyone->sol/medium reset
    first, which discarded exactly the selections the Astra swap exists to carry over.
    The swap now goes first and the reset stands down."""
    temp_db.conn.execute("ALTER TABLE user_preferences DROP COLUMN gpt56_migrated")
    _drop_gpt6_sentinel(temp_db)
    _insert_user(temp_db, "U1", "gpt-5.6-sol", "high")
    _insert_user(temp_db, "U2", "gpt-5.5", "xhigh")
    _insert_user(temp_db, "U3", "gpt-5.6-luna", "none")
    temp_db._run_migrations()

    rows = {
        r["slack_user_id"]: (r["model"], r["reasoning_effort"])
        for r in temp_db.conn.execute(
            "SELECT slack_user_id, model, reasoning_effort FROM user_preferences")
    }
    assert rows == {
        "U1": ("gpt-6-astra", "high"),
        "U2": ("gpt-6-astra", "xhigh"),
        "U3": ("gpt-6-astra", "low"),   # `none` is off Astra's ladder
    }

    columns = _columns(temp_db, "user_preferences")
    assert 'gpt6_migrated' in columns
    assert 'gpt56_migrated' in columns
