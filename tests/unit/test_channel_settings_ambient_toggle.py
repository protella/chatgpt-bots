"""F51 UI — the per-channel ambient-memory opt-out toggle in the channel settings modal.

The opt-out already exists end-to-end (DB column + capture/backfill/fetch_url enforcement); this
covers the only missing piece, the Slack modal control. The checkbox is injected into the modal by
the settings handler (the shared builder in settings_modal.py is untouched), so these tests drive
the real open / model-change / submit handlers.

Contract:
  - Checked = capturing (the default). Stored None (inherit) or True → box ticked.
  - Unchecked = opted out. Stored False → box empty; submit persists False.
  - Checking it back on clears to NULL (inherit) — it never freezes an explicit row.
  - The global master switch (ENABLE_AMBIENT_MEMORY) hides the toggle when off, and a save then
    never writes/clobbers the stored value.

All I/O is stubbed or backed by a throwaway SQLite file. No live Slack, no API.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from config import config
from database import DatabaseManager
from settings_modal import SettingsModal
from slack_client.event_handlers.settings import SlackSettingsHandlersMixin


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


def _make_host(db):
    host = SlackSettingsHandlersMixin.__new__(SlackSettingsHandlersMixin)
    host.app = _FakeApp()
    host.db = db
    host.settings_modal = SettingsModal.__new__(SettingsModal)
    host.log_info = host.log_error = host.log_debug = host.log_warning = lambda *a, **k: None
    host._register_settings_handlers()
    return host


def _stub_db(row):
    return SimpleNamespace(
        get_channel_settings_async=AsyncMock(return_value=row),
        get_channel_memory_async=AsyncMock(return_value=[]),
    )


def _state(*, model="inherit", effort="inherit", verbosity="inherit",
           participation="inherit", directives=None, placement="inherit", ambient="on"):
    """A modal state.values payload. ambient: "on" (checked), "off" (empty box), or None (block
    omitted, as when the master switch hid it)."""
    def sel(v):
        return {"selected_option": {"value": v}}
    state = {
        "channel_model_block": {"channel_model": sel(model)},
        "channel_effort_block": {"channel_reasoning_effort": sel(effort)},
        "channel_verbosity_block": {"channel_verbosity": sel(verbosity)},
        "participation_block": {"participation_level": sel(participation)},
        "directives_block": {"directives": {"value": directives}},
        "reply_in_channel_block": {"reply_in_channel": sel(placement)},
    }
    if ambient is not None:
        opts = [{"value": "on"}] if ambient == "on" else []
        state["ambient_memory_block"] = {"ambient_memory": {"selected_options": opts}}
    return state


def _view(state, channel_id="C1"):
    return {"id": "V1",
            "private_metadata": json.dumps({"channel_id": channel_id}),
            "state": {"values": state}}


def _ambient_block(view):
    return next((b for b in view["blocks"] if b.get("block_id") == "ambient_memory_block"), None)


def _is_checked(block):
    return "initial_options" in block["element"]


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


@pytest.fixture(autouse=True)
def _ambient_on(monkeypatch):
    # Default the master switch ON for every test; the "master off" cases flip it explicitly.
    monkeypatch.setattr(config, "enable_ambient_memory", True)


async def _open(host, client, channel_id="C1"):
    handler = host.app.actions["open_channel_settings"]
    body = {"trigger_id": "T", "container": {"channel_id": channel_id}, "user": {"id": "U1"}}
    await handler(ack=AsyncMock(), body=body, client=client)
    return client.views_open.await_args.kwargs["view"]


async def _submit(host, state, client, channel_id="C1"):
    handler = host.app.views["channel_settings_modal"]
    body = {"user": {"id": "U1"}, "view": _view(state, channel_id)}
    await handler(ack=AsyncMock(), body=body, view=body["view"], client=client)


# --------------------------------------------------------------------------- open: toggle rendered
@pytest.mark.asyncio
class TestModalShowsToggle:
    async def test_no_row_shows_checked(self):
        # No stored row → inherit → capturing → box ticked.
        host = _make_host(_stub_db(None))
        client = SimpleNamespace(views_open=AsyncMock())
        view = await _open(host, client)
        block = _ambient_block(view)
        assert block is not None
        assert _is_checked(block)
        assert block["optional"] is True
        assert block["element"]["action_id"] == "ambient_memory"

    async def test_inherit_none_shows_checked(self):
        host = _make_host(_stub_db({"ambient_memory": None}))
        view = await _open(host, SimpleNamespace(views_open=AsyncMock()))
        assert _is_checked(_ambient_block(view))

    async def test_explicit_true_shows_checked(self):
        host = _make_host(_stub_db({"ambient_memory": True}))
        view = await _open(host, SimpleNamespace(views_open=AsyncMock()))
        assert _is_checked(_ambient_block(view))

    async def test_explicit_false_shows_unchecked(self):
        host = _make_host(_stub_db({"ambient_memory": False}))
        view = await _open(host, SimpleNamespace(views_open=AsyncMock()))
        assert not _is_checked(_ambient_block(view))

    async def test_master_off_hides_toggle(self, monkeypatch):
        monkeypatch.setattr(config, "enable_ambient_memory", False)
        host = _make_host(_stub_db({"ambient_memory": False}))
        view = await _open(host, SimpleNamespace(views_open=AsyncMock()))
        assert _ambient_block(view) is None


# --------------------------------------------------------------------------- submit persistence
@pytest.mark.asyncio
class TestSubmitPersists:
    async def test_uncheck_persists_opt_out(self, temp_db):
        host = _make_host(temp_db)
        client = SimpleNamespace(chat_postEphemeral=AsyncMock())
        await _submit(host, _state(ambient="off"), client)
        assert (await temp_db.get_channel_settings_async("C1"))["ambient_memory"] is False

    async def test_check_clears_prior_opt_out(self, temp_db):
        # Channel was opted out; re-checking the box turns capture back on (→ NULL/inherit).
        await temp_db.set_channel_settings_async("C1", ambient_memory=False)
        assert (await temp_db.get_channel_settings_async("C1"))["ambient_memory"] is False
        host = _make_host(temp_db)
        await _submit(host, _state(ambient="on"), SimpleNamespace(chat_postEphemeral=AsyncMock()))
        # None (inherit) — not opted out. Enforcement only trips on an explicit False.
        assert (await temp_db.get_channel_settings_async("C1"))["ambient_memory"] is None

    async def test_opt_out_preserves_other_fields(self, temp_db):
        await temp_db.set_channel_settings_async("C1", participation_level="active")
        host = _make_host(temp_db)
        await _submit(host, _state(participation="active", ambient="off"),
                      SimpleNamespace(chat_postEphemeral=AsyncMock()))
        row = await temp_db.get_channel_settings_async("C1")
        assert row["ambient_memory"] is False
        assert row["participation_level"] == "active"

    async def test_master_off_does_not_write(self, temp_db, monkeypatch):
        # Master switch off: even if a stale block rode in, submit must not touch ambient_memory.
        await temp_db.set_channel_settings_async("C1", ambient_memory=False)
        monkeypatch.setattr(config, "enable_ambient_memory", False)
        host = _make_host(temp_db)
        await _submit(host, _state(ambient="on"), SimpleNamespace(chat_postEphemeral=AsyncMock()))
        # Untouched — the opt-out stored earlier survives.
        assert (await temp_db.get_channel_settings_async("C1"))["ambient_memory"] is False


# --------------------------------------------------------------------------- overlay round-trip
class TestOverlayRoundTrip:
    def _overlay(self, row, **state_kw):
        return SlackSettingsHandlersMixin._overlay_channel_form_state(row, _state(**state_kw))

    def test_checked_overlays_on(self):
        assert self._overlay({"ambient_memory": False}, ambient="on")["ambient_memory"] is None

    def test_unchecked_overlays_off(self):
        assert self._overlay({}, ambient="off")["ambient_memory"] is False

    def test_absent_block_leaves_stored_value(self):
        # Master switch off → block not rendered → the stored value must survive a re-render.
        assert self._overlay({"ambient_memory": False}, ambient=None)["ambient_memory"] is False


# --------------------------------------------------------------------------- model-change re-render
@pytest.mark.asyncio
class TestRerenderKeepsToggle:
    async def test_unchecked_survives_model_change(self):
        host = _make_host(_stub_db({"ambient_memory": None, "model": None,
                                    "reasoning_effort": None, "verbosity": None}))
        client = SimpleNamespace(views_update=AsyncMock())
        # User unticked the box, then switched the model — the empty box must survive the rebuild.
        state = _state(model="gpt-5.5", ambient="off")
        body = {"user": {"id": "U1"}, "view": _view(state),
                "actions": [{"action_id": "channel_model", "value": None}]}
        await host.app.actions["channel_model"](ack=AsyncMock(), body=body, client=client)
        rebuilt = client.views_update.await_args.kwargs["view"]
        assert not _is_checked(_ambient_block(rebuilt))
