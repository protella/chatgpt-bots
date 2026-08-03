"""Channel capability profile + the DM/channel surface discriminator (P1, spec §3b/§8).

The rule under test: on a channel turn, what MACHINE answers is the channel's business, not the
requester's. Two people asking the same channel the same question must get the same model,
effort, and tool capabilities — otherwise a shared stream's answers depend on who spoke last.
DM turns are untouched, and that is asserted explicitly.
"""
import pytest

import config as config_module
from config import BotConfig, CHANNEL_CAPABILITY_KEYS, clamp_effort
from slack_client.utilities import is_dm_conversation

COSMETIC_IMAGE_KEYS = ("image_size", "image_quality", "image_background", "image_format",
                       "image_compression", "input_fidelity")


class FakeDB:
    """Minimal stand-in for the two reads get_thread_config performs."""

    def __init__(self, prefs=None, channel_settings=None):
        self._prefs = prefs or {}
        self._channel_settings = channel_settings

    def get_user_preferences(self, user_id):
        return self._prefs.get(user_id)

    async def get_user_preferences_async(self, user_id):
        return self._prefs.get(user_id)

    def get_channel_settings(self, channel_id):
        return self._channel_settings

    async def get_channel_settings_async(self, channel_id):
        return self._channel_settings


ALICE = {
    "model": "gpt-5.5",
    "reasoning_effort": "xhigh",
    "verbosity": "high",
    "enable_web_search": False,
    "enable_mcp": False,
    "image_model": "gpt-image-1",
    "image_size": "1024x1024",
    "image_quality": "high",
    "temperature": 0.3,
    "custom_instructions": "be terse",
}
BOB = {
    "model": "gpt-5.6-terra",
    "reasoning_effort": "none",
    "verbosity": "low",
    "enable_web_search": True,
    "enable_mcp": True,
    "image_model": "gpt-image-2",
    "image_size": "1536x1024",
    "image_quality": "low",
}


@pytest.fixture
def cfg():
    return BotConfig()


def _db(channel_settings=None):
    return FakeDB({"U_ALICE": ALICE, "U_BOB": BOB}, channel_settings)


# --------------------------------------------------------------------- the key list

def test_capability_keys_are_exactly_the_approved_set():
    assert CHANNEL_CAPABILITY_KEYS == (
        "model", "reasoning_effort", "verbosity", "enable_web_search", "enable_mcp",
        "image_model", "enable_code_interpreter")


# --------------------------------------------------------------------- DM path unchanged

def test_dm_path_is_the_documented_legacy_hierarchy(cfg):
    """defaults <- user prefs <- channel shared settings <- thread overrides, clamped."""
    channel_settings = {"model": "gpt-5.6-luna", "reasoning_effort": None, "verbosity": None}
    overrides = {"verbosity": "medium"}

    expected = cfg._default_thread_config()
    expected.update(cfg._map_user_prefs(ALICE))
    expected.update(cfg._map_channel_settings(channel_settings))
    expected.update(overrides)
    expected["reasoning_effort"] = clamp_effort(expected["model"],
                                                expected["reasoning_effort"])

    got = cfg.get_thread_config(overrides=dict(overrides), user_id="U_ALICE",
                                db=_db(channel_settings), channel_id="C1")
    assert got == expected


def test_dm_default_is_off_and_keeps_requester_capabilities(cfg):
    dm = cfg.get_thread_config(user_id="U_ALICE", db=_db(), channel_id="D1")
    assert dm["model"] == "gpt-5.5"
    assert dm["enable_web_search"] is False
    assert dm["image_model"] == "gpt-image-1"
    # Explicitly passing the default must not change a thing.
    assert dm == cfg.get_thread_config(user_id="U_ALICE", db=_db(), channel_id="D1",
                                       channel_turn=False)


@pytest.mark.parametrize("user_id,prefs", [("U_ALICE", ALICE), ("U_BOB", BOB)])
def test_dm_keeps_every_requester_capability(cfg, user_id, prefs):
    """The one new thing on the DM path is the stripping — so assert it never happens."""
    got = cfg.get_thread_config(user_id=user_id, db=_db(), channel_id="D1")
    for key in CHANNEL_CAPABILITY_KEYS:
        if key not in prefs:
            continue
        expected = (clamp_effort(prefs["model"], prefs[key])
                    if key == "reasoning_effort" else prefs[key])
        assert got[key] == expected, key


def test_two_dm_requesters_still_differ(cfg):
    alice = cfg.get_thread_config(user_id="U_ALICE", db=_db(), channel_id="D1")
    bob = cfg.get_thread_config(user_id="U_BOB", db=_db(), channel_id="D2")
    assert alice["model"] != bob["model"]


# --------------------------------------------------------------------- channel turns

def test_two_requesters_resolve_to_identical_capabilities(cfg):
    db = _db({"model": "gpt-5.6-luna", "reasoning_effort": "low", "verbosity": None})
    alice = cfg.get_thread_config(user_id="U_ALICE", db=db, channel_id="C1", channel_turn=True)
    bob = cfg.get_thread_config(user_id="U_BOB", db=db, channel_id="C1", channel_turn=True)

    for key in CHANNEL_CAPABILITY_KEYS:
        assert alice[key] == bob[key], key
    assert alice["model"] == "gpt-5.6-luna"
    assert alice["reasoning_effort"] == "low"
    assert alice["verbosity"] == cfg.default_verbosity


def test_thread_overrides_cannot_reclaim_a_capability(cfg):
    overrides = {"model": "gpt-5.5", "enable_web_search": False, "temperature": 0.9}
    got = cfg.get_thread_config(overrides=overrides, user_id="U_ALICE", db=_db(),
                                channel_id="C1", channel_turn=True)
    assert got["model"] == cfg.gpt_model
    assert got["enable_web_search"] == cfg.enable_web_search
    # Non-capability overrides are untouched.
    assert got["temperature"] == 0.9


def test_differing_thread_overrides_still_produce_one_profile(cfg):
    db = _db({"model": "gpt-5.6-terra"})
    first = cfg.get_thread_config(overrides={"model": "gpt-5.5", "verbosity": "high"},
                                  user_id="U_ALICE", db=db, channel_id="C1", channel_turn=True)
    second = cfg.get_thread_config(overrides={"reasoning_effort": "max", "enable_mcp": False},
                                   user_id="U_BOB", db=db, channel_id="C1", channel_turn=True)
    assert {k: first[k] for k in CHANNEL_CAPABILITY_KEYS} == \
           {k: second[k] for k in CHANNEL_CAPABILITY_KEYS}


def test_channel_settings_beat_globals(cfg):
    db = _db({"model": "gpt-5.5", "reasoning_effort": "xhigh", "verbosity": "low"})
    got = cfg.get_thread_config(user_id="U_ALICE", db=db, channel_id="C1", channel_turn=True)
    assert (got["model"], got["reasoning_effort"], got["verbosity"]) == (
        "gpt-5.5", "xhigh", "low")


def test_unset_channel_columns_fall_through_to_globals(cfg):
    db = _db({"model": None, "reasoning_effort": None, "verbosity": None})
    got = cfg.get_thread_config(user_id="U_ALICE", db=db, channel_id="C1", channel_turn=True)
    assert got["model"] == cfg.gpt_model
    assert got["reasoning_effort"] == clamp_effort(cfg.gpt_model, cfg.default_reasoning_effort)
    assert got["verbosity"] == cfg.default_verbosity


def test_explicit_falsy_channel_value_survives(cfg):
    """`is not None`, not truthiness — a deliberately OFF setting is still a decision.

    Read against a boolean rather than a vocabulary column: respec §6.2 gave `verbosity` an
    allowlist, so an empty string there is no longer an explicit decision but an unusable value
    that falls back. `enable_web_search = 0` is the case the rule exists for — truthiness would
    read the 0 as "unset" and hand the channel back the capability it switched off.
    """
    profile = cfg._channel_capability_profile({"model": "gpt-5.5", "enable_web_search": 0,
                                               "reasoning_effort": None})
    assert profile["enable_web_search"] is False
    assert profile["model"] == "gpt-5.5"
    assert profile["reasoning_effort"] == clamp_effort("gpt-5.5", cfg.default_reasoning_effort)


def test_globally_disabled_capability_stays_off_for_an_eager_requester(cfg, monkeypatch):
    monkeypatch.setattr(cfg, "enable_web_search", False)
    monkeypatch.setattr(cfg, "enable_code_interpreter", False)
    got = cfg.get_thread_config(user_id="U_BOB", db=_db(), channel_id="C1", channel_turn=True)
    assert got["enable_web_search"] is False
    assert got["enable_code_interpreter"] is False


def test_capability_keys_are_all_present_on_a_channel_turn(cfg):
    got = cfg.get_thread_config(user_id="U_ALICE", db=_db(), channel_id="C1", channel_turn=True)
    assert all(key in got for key in CHANNEL_CAPABILITY_KEYS)


def test_cosmetic_image_prefs_stay_with_the_requester(cfg):
    alice = cfg.get_thread_config(user_id="U_ALICE", db=_db(), channel_id="C1",
                                  channel_turn=True)
    bob = cfg.get_thread_config(user_id="U_BOB", db=_db(), channel_id="C1", channel_turn=True)
    assert alice["image_size"] == "1024x1024"
    assert bob["image_size"] == "1536x1024"
    assert alice["image_model"] == bob["image_model"]
    assert any(alice[k] != bob[k] for k in COSMETIC_IMAGE_KEYS)


def test_custom_instructions_are_not_a_capability(cfg):
    got = cfg.get_thread_config(user_id="U_ALICE", db=_db(), channel_id="C1", channel_turn=True)
    assert got["custom_instructions"] == "be terse"


def test_effort_is_clamped_against_the_channel_model(cfg, monkeypatch):
    """The effort a channel runs on is always legal for the model that channel resolved to.

    respec §6.2: a stored effort outside the resolved model's ladder is not silently nudged to
    the nearest legal rung — it is refused, and the GLOBAL default takes its place, clamped
    against that same model. Nudging would leave the resolver claiming an effort the settings
    modal already renders as "inherit"; falling back makes the two agree.
    """
    db = _db({"model": "gpt-5.5", "reasoning_effort": "max"})
    got = cfg.get_thread_config(user_id="U_ALICE", db=db, channel_id="C1", channel_turn=True)
    assert got["reasoning_effort"] == clamp_effort("gpt-5.5", cfg.default_reasoning_effort)

    # And the fallback itself is clamped, never a literal: a global default of `max` is not a
    # thing gpt-5.5 accepts either.
    monkeypatch.setattr(cfg, "default_reasoning_effort", "max")
    got = cfg.get_thread_config(user_id="U_ALICE", db=db, channel_id="C2", channel_turn=True)
    assert got["reasoning_effort"] == "xhigh"


async def test_async_twin_matches_the_sync_one(cfg):
    db = _db({"model": "gpt-5.6-luna"})
    sync = cfg.get_thread_config(user_id="U_ALICE", db=db, channel_id="C1", channel_turn=True)
    got = await cfg.get_thread_config_async(user_id="U_ALICE", db=db, channel_id="C1",
                                            channel_turn=True)
    assert got == sync

    dm_sync = cfg.get_thread_config(user_id="U_ALICE", db=db, channel_id="D1")
    dm_async = await cfg.get_thread_config_async(user_id="U_ALICE", db=db, channel_id="D1")
    assert dm_async == dm_sync


def test_no_prefs_and_no_channel_row_is_pure_globals(cfg):
    got = cfg.get_thread_config(db=FakeDB(), channel_id="C1", channel_turn=True)
    assert got["model"] == cfg.gpt_model
    assert got["enable_mcp"] == cfg.mcp_enabled_default
    assert got["image_model"] == cfg.image_model


# --------------------------------------------------------------------- surface ruling

@pytest.mark.parametrize("channel_id,channel_type,expected", [
    ("D0123", None, True),
    ("D0123", "im", True),
    ("U0123", None, True),          # outbound DMs are posted with channel=<user_id>
    ("U0123", "im", True),
    ("C0123", None, False),
    ("C0123", "channel", False),
    ("G0123", None, False),
    ("G0123", "group", False),
    ("C0123", "mpim", False),       # MPIMs already route through the channel path
    ("G0123", "mpim", False),
    ("C0123", "im", True),          # an explicit type beats the prefix
    ("D0123", "channel", False),
    ("X0123", None, True),          # unknown → DM side, fail-safe for receipts
    ("", None, True),
    (None, None, True),
])
def test_is_dm_conversation_truth_table(channel_id, channel_type, expected):
    assert is_dm_conversation(channel_id, channel_type) is expected


def test_is_dm_conversation_defaults_to_prefix_only():
    assert is_dm_conversation("C0123") is False
    assert is_dm_conversation("D0123") is True


# ------------------------------------------------- W4: the three new capability columns (§6.1/6.2)


@pytest.fixture
def quiet_reporter(monkeypatch):
    """A fresh warned-set per test.

    `_REPORTED_UNUSABLE_CHANNEL_SETTINGS` is process-local and never cleared in production — one
    bad row must not write the same warning on every turn forever. A test that counts warnings
    therefore has to start from empty, or the count depends on which test ran first.
    """
    monkeypatch.setattr(config_module, "_REPORTED_UNUSABLE_CHANNEL_SETTINGS", set())


def test_the_resolver_honours_the_three_new_columns(cfg, quiet_reporter, monkeypatch):
    """T102. Each new column overrides its global default; an explicit False stays False; NULL
    inherits; and a value outside the column's allowlist resolves to the global default rather
    than to whatever `bool()` would have made of it."""
    profile = cfg._channel_capability_profile(
        {"enable_web_search": 0, "enable_mcp": 1, "image_model": "gpt-image-1",
         "verbosity": "high"})
    assert profile["enable_web_search"] is False
    assert profile["enable_mcp"] is True
    assert profile["image_model"] == "gpt-image-1"
    assert profile["verbosity"] == "high"

    # NULL is inherit, for all three.
    inherited = cfg._channel_capability_profile(
        {"enable_web_search": None, "enable_mcp": None, "image_model": None})
    assert inherited["enable_web_search"] == cfg.enable_web_search
    assert inherited["enable_mcp"] == cfg.mcp_enabled_default
    assert inherited["image_model"] == cfg.image_model

    # Verbosity is a vocabulary, not free text.
    assert cfg._channel_capability_profile({"verbosity": "chatty"})["verbosity"] == \
        cfg.default_verbosity

    # `2` and `-1` are reachable only on a row written before the CHECK constraint existed.
    # `bool(2)` and `bool(-1)` are both True, which would switch a capability ON for a channel
    # that never asked — so the resolver refuses the value outright.
    #
    # Read against globals that are OFF, deliberately. With them on, "resolves to the global
    # default" and "resolves to True" name the same value, and no assertion here could tell a
    # working resolver from one that just called `bool()` on whatever was stored.
    monkeypatch.setattr(cfg, "enable_web_search", False)
    monkeypatch.setattr(cfg, "mcp_enabled_default", False)
    for rogue in (2, -1):
        got = cfg._channel_capability_profile({"enable_web_search": rogue, "enable_mcp": rogue})
        assert got["enable_web_search"] is False
        assert got["enable_mcp"] is False


def test_an_illegal_stored_value_falls_back_loudly(cfg, quiet_reporter, caplog):
    """T106. An unlisted model / image model / effort resolves to the global default and says so,
    naming the channel — and the turn still runs, because every fallback target is a value
    `validate()` already vouched for at boot."""
    with caplog.at_level("WARNING", logger="bot.config"):
        profile = cfg._channel_capability_profile(
            {"model": "gpt-4o", "image_model": "gpt-image-9", "reasoning_effort": "ludicrous"},
            "C0BKX77NU66")

    assert profile["model"] == cfg.gpt_model
    assert profile["image_model"] == cfg.image_model
    assert profile["reasoning_effort"] == clamp_effort(cfg.gpt_model,
                                                       cfg.default_reasoning_effort)
    warned = "\n".join(r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    assert "C0BKX77NU66" in warned
    for column in ("model", "image_model", "reasoning_effort"):
        assert column in warned


def test_the_invalid_value_warning_fires_once(cfg, quiet_reporter, caplog):
    """T105. Ten consecutive resolutions of the same bad row log ONE warning, not ten — the
    resolver runs on every channel turn, and an unbounded warning would fill the log forever."""
    with caplog.at_level("WARNING", logger="bot.config"):
        for _ in range(10):
            cfg._channel_capability_profile({"image_model": "gpt-image-9"}, "C1")
    bad_image = [r for r in caplog.records
                 if r.levelname == "WARNING" and "image_model" in r.getMessage()]
    assert len(bad_image) == 1

    # …and it is bounded per (channel, column), not globally: another channel with the same bad
    # value is a different operator with a different row to go fix.
    with caplog.at_level("WARNING", logger="bot.config"):
        cfg._channel_capability_profile({"image_model": "gpt-image-9"}, "C2")
    assert len([r for r in caplog.records
                if r.levelname == "WARNING" and "image_model" in r.getMessage()]) == 2


def test_the_tri_state_checks_reject_out_of_range(tmp_path, monkeypatch):
    """T103. Against real SQLite: the two boolean columns accept NULL, 0 and 1 and nothing else.
    Without the CHECK, SQLite would happily store `2` in an INTEGER column."""
    import sqlite3

    from database import DatabaseManager

    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="tristate")
    try:
        for column in ("enable_web_search", "enable_mcp"):
            for legal in (None, 0, 1):
                db.conn.execute(
                    f"INSERT INTO channel_settings (channel_id, {column}) VALUES (?, ?) "
                    f"ON CONFLICT(channel_id) DO UPDATE SET {column} = excluded.{column}",
                    (f"C_{column}", legal))
            for rogue in (2, -1):
                with pytest.raises(sqlite3.IntegrityError):
                    db.conn.execute(
                        f"INSERT INTO channel_settings (channel_id, {column}) VALUES (?, ?)",
                        (f"C_rogue_{column}_{rogue}", rogue))
    finally:
        db.conn.close()
