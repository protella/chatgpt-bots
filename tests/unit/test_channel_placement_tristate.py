"""Round-2 fixes for the participation-backoff redesign.

SHOULD-FIX #5 — the channel-settings modal must not freeze reply_in_channel inheritance. Placement
is a tri-state control; opening + saving an inheriting channel untouched has to leave the row NULL
(still inheriting), and each explicit option round-trips None / True / False through the submit path.

SHOULD-FIX #7 — a modal re-render (model-change) must preserve the user's in-flight form edits: a
cleared directives box stays cleared, and an edited participation level / placement survives
instead of being silently reverted to the stored row. (The per-row delete-memory / clear-mute
re-render buttons were removed with the mute mechanism and the per-row memory controls.)

All I/O is stubbed or backed by a throwaway SQLite file. No live Slack, no API.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from database import DatabaseManager
from settings_modal import SettingsModal


# --------------------------------------------------------------------------- fakes / builders
class _FakeApp:
    """Captures handlers registered via @app.action / @app.view (others are no-ops)."""

    def __init__(self):
        self.actions = {}
        self.views = {}

    def action(self, action_id):
        def deco(fn):
            self.actions[action_id] = fn
            return fn
        return deco

    def view(self, callback_id):
        def deco(fn):
            self.views[callback_id] = fn
            return fn
        return deco

    def command(self, *_a, **_k):
        return lambda fn: fn

    def shortcut(self, *_a, **_k):
        return lambda fn: fn

    def event(self, *_a, **_k):
        return lambda fn: fn


def _make_host(db, *, real_modal=False):
    from slack_client.event_handlers.settings import SlackSettingsHandlersMixin

    host = SlackSettingsHandlersMixin.__new__(SlackSettingsHandlersMixin)
    host.app = _FakeApp()
    host.db = db
    host.settings_modal = (SettingsModal.__new__(SettingsModal) if real_modal
                           else SimpleNamespace(build_channel_settings_modal=lambda *a, **k: {"type": "modal"}))
    host.log_info = host.log_error = host.log_debug = host.log_warning = lambda *a, **k: None
    host._register_settings_handlers()
    return host


def _state(*, model="inherit", effort="inherit", verbosity="inherit",
           participation="inherit", directives=None, placement="inherit"):
    """A modal state.values payload with every channel-settings input block populated."""
    def sel(v):
        return {"selected_option": {"value": v}}
    return {
        "channel_model_block": {"channel_model": sel(model)},
        "channel_effort_block": {"channel_reasoning_effort": sel(effort)},
        "channel_verbosity_block": {"channel_verbosity": sel(verbosity)},
        "participation_block": {"participation_level": sel(participation)},
        "directives_block": {"directives": {"value": directives}},
        "reply_in_channel_block": {"reply_in_channel": sel(placement)},
    }


def _view(state, channel_id="C1"):
    return {"id": "V1",
            "private_metadata": json.dumps({"channel_id": channel_id}),
            "state": {"values": state}}


def _block(view, block_id):
    return next(b for b in view["blocks"] if b.get("block_id") == block_id)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("os.makedirs"):
            db = DatabaseManager("test")
            db.db_path = f"{tmpdir}/test.db"
            if getattr(db, "conn", None):
                db.conn.close()
            db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
            db.conn.row_factory = sqlite3.Row
            db.conn.execute("PRAGMA journal_mode=WAL")
            db.init_schema()
            yield db
            if getattr(db, "conn", None):
                db.conn.close()


async def _submit(host, state, client, channel_id="C1"):
    handler = host.app.views["channel_settings_modal"]
    body = {"user": {"id": "U1"}, "view": _view(state, channel_id)}
    await handler(ack=AsyncMock(), body=body, view=body["view"], client=client)


# --------------------------------------------------------------------------- SHOULD-FIX #5: submit
@pytest.mark.asyncio
class TestPlacementSubmit:
    async def test_untouched_null_channel_stays_null(self, temp_db):
        # An inheriting channel (reply_in_channel NULL) that the user opens and saves WITHOUT
        # touching placement must remain NULL — never frozen into an explicit True/False row.
        await temp_db.set_channel_settings_async("C1", participation_level="active")
        before = await temp_db.get_channel_settings_async("C1")
        assert before["reply_in_channel"] is None  # precondition: inheriting

        host = _make_host(temp_db)
        client = SimpleNamespace(chat_postEphemeral=AsyncMock())
        # Modal reflects the stored row: participation 'active', placement 'inherit' (untouched).
        await _submit(host, _state(participation="active", placement="inherit"), client)

        row = await temp_db.get_channel_settings_async("C1")
        assert row["reply_in_channel"] is None          # STILL inheriting, not frozen
        assert row["participation_level"] == "active"   # untouched fields preserved

    async def test_channel_option_writes_true(self, temp_db):
        host = _make_host(temp_db)
        client = SimpleNamespace(chat_postEphemeral=AsyncMock())
        await _submit(host, _state(placement="channel"), client)
        assert (await temp_db.get_channel_settings_async("C1"))["reply_in_channel"] is True

    async def test_threads_option_writes_false(self, temp_db):
        host = _make_host(temp_db)
        client = SimpleNamespace(chat_postEphemeral=AsyncMock())
        await _submit(host, _state(placement="threads"), client)
        assert (await temp_db.get_channel_settings_async("C1"))["reply_in_channel"] is False

    async def test_inherit_option_clears_an_explicit_row(self, temp_db):
        # An explicit placement can be RESET to inherit (→ NULL) from the modal.
        await temp_db.set_channel_settings_async("C1", reply_in_channel=True)
        assert (await temp_db.get_channel_settings_async("C1"))["reply_in_channel"] is True
        host = _make_host(temp_db)
        client = SimpleNamespace(chat_postEphemeral=AsyncMock())
        await _submit(host, _state(placement="inherit"), client)
        assert (await temp_db.get_channel_settings_async("C1"))["reply_in_channel"] is None


# --------------------------------------------------------------------------- SHOULD-FIX #7: overlay helper
class TestOverlayInFlightState:
    """Unit-tests the shared re-render merge: in-flight form values are authoritative."""

    def _overlay(self, row, **state_kw):
        from slack_client.event_handlers.settings import SlackSettingsHandlersMixin
        return SlackSettingsHandlersMixin._overlay_channel_form_state(row, _state(**state_kw))

    def test_cleared_directives_stay_cleared(self):
        # User cleared the box → must NOT be restored from the stored row.
        assert self._overlay({"directives": "stored rule"}, directives=None)["directives"] is None

    def test_edited_directives_win(self):
        assert self._overlay({"directives": "old"}, directives="new")["directives"] == "new"

    def test_whitespace_only_directives_treated_as_cleared(self):
        assert self._overlay({"directives": "old"}, directives="   ")["directives"] is None

    def test_inherit_participation_clears_stale_response_mode(self):
        # Stored an explicit mode; user moved the control to 'inherit'. BOTH columns must clear so
        # the builder's legacy fallback can't resurrect the stored mode and reselect the old level.
        cs = self._overlay({"participation_level": "active", "response_mode": "auto_respond"},
                           participation="inherit")
        assert cs["participation_level"] is None
        assert cs["response_mode"] is None

    def test_edited_participation_wins_and_syncs_mode(self):
        cs = self._overlay({"participation_level": "mentions_only", "response_mode": "tag_only"},
                           participation="active")
        assert cs["participation_level"] == "active"
        # response_mode kept in lockstep (LEVEL_TO_MODE), not left stale.
        from message_processor.participation import LEVEL_TO_MODE
        assert cs["response_mode"] == LEVEL_TO_MODE.get("active")

    def test_placement_tristate_roundtrip(self):
        assert self._overlay({}, placement="inherit")["reply_in_channel"] is None
        assert self._overlay({}, placement="channel")["reply_in_channel"] is True
        assert self._overlay({}, placement="threads")["reply_in_channel"] is False

    def test_placement_inherit_wins_over_stored_true(self):
        assert self._overlay({"reply_in_channel": True}, placement="inherit")["reply_in_channel"] is None


# --------------------------------------------------------------------------- SHOULD-FIX #7: end-to-end re-render
_STORED = {"participation_level": "mentions_only", "response_mode": "tag_only",
           "directives": "stored rule", "reply_in_channel": True,
           "model": None, "reasoning_effort": None, "verbosity": None}

# In-flight edits the user made but has NOT saved when they click a re-render button.
_EDITED = dict(participation="active", directives=None, placement="inherit")


def _assert_edits_survived(rebuilt):
    """The rebuilt modal must show the in-flight edits, not the stored row."""
    assert _block(rebuilt, "participation_block")["element"]["initial_option"]["value"] == "active"
    assert _block(rebuilt, "directives_block")["element"]["initial_value"] == ""      # cleared, not restored
    assert _block(rebuilt, "reply_in_channel_block")["element"]["initial_option"]["value"] == "inherit"


@pytest.mark.asyncio
class TestRerenderPreservesEdits:
    def _host(self, memory_rows=None):
        db = SimpleNamespace(
            get_channel_settings_async=AsyncMock(return_value=dict(_STORED)),
            get_channel_memory_async=AsyncMock(return_value=memory_rows or []),
        )
        return _make_host(db, real_modal=True)

    async def test_model_change_rerender_keeps_edits(self):
        host = self._host()
        client = SimpleNamespace(views_update=AsyncMock())
        # Model swapped to gpt-5.5; participation/directives/placement edits must survive the rebuild.
        state = _state(model="gpt-5.5", **_EDITED)
        body = {"user": {"id": "U1"}, "view": _view(state),
                "actions": [{"action_id": "channel_model", "value": None}]}
        await host.app.actions["channel_model"](ack=AsyncMock(), body=body, client=client)
        rebuilt = client.views_update.await_args.kwargs["view"]
        _assert_edits_survived(rebuilt)
        # ...and the model itself reflects the new selection.
        assert _block(rebuilt, "channel_model_block")["element"]["initial_option"]["value"] == "gpt-5.5"
