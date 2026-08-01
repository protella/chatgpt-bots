"""Channel capability profile + the DM/channel surface discriminator (P1, spec §3b/§8).

The rule under test: on a channel turn, what MACHINE answers is the channel's business, not the
requester's. Two people asking the same channel the same question must get the same model,
effort, and tool capabilities — otherwise a shared stream's answers depend on who spoke last.
DM turns are untouched, and that is asserted explicitly.
"""
import pytest

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
    """`is not None`, not truthiness — a deliberately empty setting is still a decision."""
    profile = cfg._channel_capability_profile({"model": "gpt-5.5", "verbosity": "",
                                               "reasoning_effort": None})
    assert profile["verbosity"] == ""
    assert profile["model"] == "gpt-5.5"
    assert profile["reasoning_effort"] == cfg.default_reasoning_effort


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


def test_effort_is_clamped_against_the_channel_model(cfg):
    db = _db({"model": "gpt-5.5", "reasoning_effort": "max"})
    got = cfg.get_thread_config(user_id="U_ALICE", db=db, channel_id="C1", channel_turn=True)
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
