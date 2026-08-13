"""Phase 7 entry point — .env-backed config lists, the channel-settings modal, and the
Configure-button footer. No slash command. All stubbed I/O — no live bot, no legacy suite.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config, _env_list
from database import DatabaseManager
from settings_modal import SettingsModal
from slack_client.messaging import SlackMessagingMixin
from base_client import Message, Response


# --------------------------------------------------------------------------- env lists
class TestEnvList:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("XLIST", raising=False)
        assert _env_list("XLIST", ["a", "b"]) == ["a", "b"]

    def test_comma_split_and_trim(self, monkeypatch):
        monkeypatch.setenv("XLIST", " a, b ,c,  ")
        assert _env_list("XLIST", ["z"]) == ["a", "b", "c"]

    def test_empty_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("XLIST", "  , ,")
        assert _env_list("XLIST", ["z"]) == ["z"]


# --------------------------------------------------------------------------- modal builder
class TestChannelSettingsModal:
    @pytest.fixture
    def modal(self):
        return SettingsModal(db=MagicMock())

    def _block(self, view, block_id):
        return next(b for b in view["blocks"] if b.get("block_id") == block_id)

    def test_no_row_inherits(self, modal):
        view = modal.build_channel_settings_modal("C1", None, "tag_only")
        assert view["callback_id"] == "channel_settings_modal"
        assert json.loads(view["private_metadata"])["channel_id"] == "C1"
        assert self._block(view, "participation_block")["element"]["initial_option"]["value"] == "inherit"
        assert self._block(view, "policy_block")["element"]["initial_value"] == ""
        # No saved row → reply placement is the tri-state "inherit" option (NOT resolved to the
        # global default), so opening + saving untouched can never freeze that default into a row.
        element = self._block(view, "reply_in_channel_block")["element"]
        assert element["type"] == "static_select"
        assert element["initial_option"]["value"] == "inherit"

    def test_prefill_from_row(self, modal):
        # Legacy row (response_mode only) maps to its participation-level equivalent.
        cs = {"response_mode": "auto_respond", "reply_in_channel": True}
        view = modal.build_channel_settings_modal(
            "C2", cs, "tag_only", channel_policy={"content": "only deploys"})
        assert self._block(view, "participation_block")["element"]["initial_option"]["value"] == "on"
        # The standing policy comes from the reserved policy ROW, never from the settings row.
        assert self._block(view, "policy_block")["element"]["initial_value"] == "only deploys"
        # reply_in_channel True → "channel"; False → "threads"; None → "inherit".
        assert self._block(view, "reply_in_channel_block")["element"]["initial_option"]["value"] == "channel"

    def test_null_mode_treated_as_inherit(self, modal):
        view = modal.build_channel_settings_modal("C3", {"response_mode": None}, "off")
        assert self._block(view, "participation_block")["element"]["initial_option"]["value"] == "inherit"


# --------------------------------------------------------------------------- footer blocks
class TestFooterBlocks:
    def test_single_compact_row(self):
        """One actions block only — a single button carrying the model name (no context row)."""
        blocks = SlackMessagingMixin._build_response_footer_blocks(MagicMock(), "gpt-5.5")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "actions"
        button = blocks[0]["elements"][0]
        assert button["action_id"] == "open_channel_settings"
        assert "gpt-5.5" in button["text"]["text"]

    def test_model_fallback(self):
        blocks = SlackMessagingMixin._build_response_footer_blocks(MagicMock(), None)
        assert config.gpt_model in blocks[0]["elements"][0]["text"]["text"]


# --------------------------------------------------------------------------- footer gating
class TestFooterPosting:
    def _fake_self(self):
        s = MagicMock()
        s.app.client.chat_postMessage = AsyncMock()
        s.log_debug = MagicMock()
        s._build_response_footer_blocks = SlackMessagingMixin._build_response_footer_blocks.__get__(s)
        return s

    @pytest.mark.asyncio
    async def test_posts_for_channel_text(self, monkeypatch):
        monkeypatch.setattr(config, "enable_response_footer", True)
        s = self._fake_self()
        msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1")
        resp = Response(type="text", content="hello", metadata={"model": "gpt-5.5"})
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, resp)
        s.app.client.chat_postMessage.assert_awaited_once()
        kwargs = s.app.client.chat_postMessage.await_args.kwargs
        assert kwargs["channel"] == "C1"
        assert kwargs["blocks"][0]["elements"][0]["action_id"] == "open_channel_settings"

    @pytest.mark.asyncio
    async def test_skips_empty_content(self, monkeypatch):
        """Reaction-only turns (empty text) post no message, so no footer either."""
        monkeypatch.setattr(config, "enable_response_footer", True)
        s = self._fake_self()
        msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1")
        resp = Response(type="text", content="", metadata={"reaction_only": True})
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, resp)
        s.app.client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dm_gets_feedback_strip_not_footer(self, monkeypatch):
        # Phase H: DMs post the native feedback strip instead of the Configure footer.
        monkeypatch.setattr(config, "enable_response_footer", True)
        monkeypatch.setattr(config, "enable_feedback_buttons", True)  # pinned: dev .env disables it
        s = self._fake_self()
        msg = Message(text="hi", user_id="U1", channel_id="D1", thread_id="T1")
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, Response(type="text", content="x"))
        s.app.client.chat_postMessage.assert_awaited_once()
        blocks = str(s.app.client.chat_postMessage.await_args.kwargs.get("blocks", ""))
        assert "response_feedback" in blocks
        assert "open_channel_settings" not in blocks
        # Phase H+: the strip also carries the "⚙️ <model>" user-settings button.
        assert "open_user_settings" in blocks

    @pytest.mark.asyncio
    async def test_skips_when_flag_off(self, monkeypatch):
        monkeypatch.setattr(config, "enable_response_footer", False)
        s = self._fake_self()
        msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1")
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, Response(type="text", content="x"))
        s.app.client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_non_text(self, monkeypatch):
        monkeypatch.setattr(config, "enable_response_footer", True)
        s = self._fake_self()
        msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1")
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, Response(type="error", content="boom"))
        s.app.client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_never_raises(self, monkeypatch):
        monkeypatch.setattr(config, "enable_response_footer", True)
        s = self._fake_self()
        s.app.client.chat_postMessage = AsyncMock(side_effect=Exception("slack down"))
        msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1")
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, Response(type="text", content="x"))  # no raise


# --------------------------------------------------------------------------- inherit → clears (what the modal submit does)
class TestInheritClears:
    @pytest.fixture
    def temp_db(self):
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

    def test_omitted_arg_preserves(self, temp_db):
        temp_db.set_channel_settings("C1", response_mode="auto_respond", verbosity="low")
        temp_db.set_channel_settings("C1", verbosity="high")  # omit mode → preserved
        assert temp_db.get_channel_settings("C1")["response_mode"] == "auto_respond"

    def test_explicit_none_clears_mode(self, temp_db):
        temp_db.set_channel_settings("C1", response_mode="auto_respond")
        temp_db.set_channel_settings("C1", response_mode=None)  # "inherit"
        assert temp_db.get_channel_settings("C1")["response_mode"] is None

    def test_explicit_none_clears_verbosity(self, temp_db):
        temp_db.set_channel_settings("C1", verbosity="low")
        temp_db.set_channel_settings("C1", verbosity=None)
        assert temp_db.get_channel_settings("C1")["verbosity"] is None

    @pytest.mark.asyncio
    async def test_async_inherit_clears(self, temp_db):
        await temp_db.set_channel_settings_async("C2", response_mode="off")
        await temp_db.set_channel_settings_async("C2", response_mode=None)
        row = await temp_db.get_channel_settings_async("C2")
        assert row["response_mode"] is None


# ------------------------------------------------------- W4: the capability profile modal (§6.3)
#
# The three capability controls (web search, MCP, image model) join the channel modal, and the
# copy that told operators the shared settings fall back to "the asker's personal preferences" is
# retired — on a channel turn they do not, and never did once §3b landed.


class _FakeApp:
    """Captures the handlers registered via @app.action / @app.view; everything else no-ops."""

    def __init__(self):
        self.views = {}

    def view(self, callback_id):
        def deco(fn):
            self.views[callback_id] = fn
            return fn
        return deco

    def action(self, *_a, **_k):
        return lambda fn: fn

    def command(self, *_a, **_k):
        return lambda fn: fn

    def shortcut(self, *_a, **_k):
        return lambda fn: fn

    def event(self, *_a, **_k):
        return lambda fn: fn


def _settings_host(db):
    """The real settings handlers, bound to a throwaway DB — no Slack, no modal builder."""
    from slack_client.event_handlers.settings import SlackSettingsHandlersMixin

    host = SlackSettingsHandlersMixin.__new__(SlackSettingsHandlersMixin)
    host.app = _FakeApp()
    host.db = db
    host.settings_modal = SettingsModal.__new__(SettingsModal)
    host.log_info = host.log_error = host.log_debug = host.log_warning = lambda *a, **k: None
    host._register_settings_handlers()
    return host


@pytest.fixture
def capability_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="capabilities")
    yield db
    db.conn.close()


def _capability_modal():
    return SettingsModal.__new__(SettingsModal)


def _inputs(view):
    """(block_id, action_id) for every static_select input, in render order."""
    return [(b["block_id"], b["element"]["action_id"]) for b in view["blocks"]
            if b.get("type") == "input" and b["element"].get("type") == "static_select"]


def test_the_channel_view_has_six_controls_and_one_model_select(monkeypatch):
    """T98. Six shared capability selects, pinned by id and order — and only ONE of them
    offers chat models. A second model control anywhere in the view would give a channel two
    places to disagree with itself about what it runs on.

    The visible labels and option text are pinned too: the owner chose this wording directly, so
    a silent edit to it is a UX change nobody approved."""
    from config import SUPPORTED_CHAT_MODELS, config as live_config

    monkeypatch.setattr(live_config, "gpt_model", "gpt-5.6-sol")
    monkeypatch.setattr(live_config, "default_reasoning_effort", "medium")
    monkeypatch.setattr(live_config, "default_verbosity", "medium")
    monkeypatch.setattr(live_config, "image_model", "gpt-image-2")
    monkeypatch.setattr(live_config, "enable_web_search", True)
    monkeypatch.setattr(live_config, "mcp_enabled_default", False)

    view = _capability_modal().build_channel_settings_modal("C1", None, "tag_only")
    assert _inputs(view) == [
        ("participation_block", "participation_level"),
        ("reply_in_channel_block", "reply_in_channel"),
        ("channel_model_block", "channel_model"),
        ("channel_effort_block", "channel_reasoning_effort"),
        ("channel_verbosity_block", "channel_verbosity"),
        ("channel_web_search_block", "channel_enable_web_search"),
        ("channel_mcp_block", "channel_enable_mcp"),
        ("channel_image_model_block", "channel_image_model"),
    ]

    by_block = {b["block_id"]: b for b in view["blocks"] if b.get("block_id")}
    offering_models = [bid for bid, block in by_block.items()
                       if block.get("type") == "input"
                       and {o["value"] for o in block["element"].get("options", [])}
                       & set(SUPPORTED_CHAT_MODELS)]
    assert offering_models == ["channel_model_block"]

    def values(block_id):
        return [o["value"] for o in by_block[block_id]["element"]["options"]]

    assert values("channel_web_search_block") == ["inherit", "on", "off"]
    assert values("channel_mcp_block") == ["inherit", "on", "off"]
    assert values("channel_image_model_block") == ["inherit", "gpt-image-2", "gpt-image-1"]
    # No stored row → all three sit on inherit rather than freezing today's global into the view.
    for block_id in ("channel_web_search_block", "channel_mcp_block",
                     "channel_image_model_block"):
        assert by_block[block_id]["element"]["initial_option"]["value"] == "inherit"

    # The owner-locked wording, verbatim: sentence-case block labels, "On"/"Off", and the same
    # full placeholder template on every one of the six.
    assert [by_block[bid]["label"]["text"] for bid in
            ("channel_model_block", "channel_effort_block", "channel_verbosity_block",
             "channel_web_search_block", "channel_mcp_block", "channel_image_model_block")] == \
        ["Model", "Reasoning effort", "Verbosity", "Web search", "MCP servers", "Image model"]

    def option_text(block_id):
        return [o["text"]["text"] for o in by_block[block_id]["element"]["options"]]

    assert option_text("channel_web_search_block")[1:] == ["On", "Off"]
    assert option_text("channel_mcp_block")[1:] == ["On", "Off"]
    assert [option_text(bid)[0] for bid in
            ("channel_model_block", "channel_effort_block", "channel_verbosity_block",
             "channel_web_search_block", "channel_mcp_block", "channel_image_model_block")] == [
        "Use the workspace default (currently: gpt-5.6-sol)",
        "Use the workspace default (currently: medium)",
        "Use the workspace default (currently: medium)",
        "Use the workspace default (currently: on)",
        "Use the workspace default (currently: off)",
        "Use the workspace default (currently: gpt-image-2)",
    ]


def test_the_channel_copy_no_longer_claims_personal_fallback():
    """T108. The retired sentence is gone and the replacement is present."""
    rendered = json.dumps(_capability_modal().build_channel_settings_modal("C1", None, "tag_only"))
    assert "asker's personal preferences" not in rendered
    assert "each person's own setting" not in rendered
    assert "These apply to everyone in this channel" in rendered
    assert "Use the workspace default (currently:" in rendered
    # The effort hint described the nearest-rung nudge the owner's ruling removed. An illegal
    # stored effort is refused and the channel takes the workspace default — the hint must not
    # claim otherwise (it now says nothing about fallback at all).
    assert "max → xhigh" not in rendered
    assert "to a nearby setting" not in rendered


@pytest.mark.asyncio
async def test_concurrent_partial_saves_do_not_clobber(capability_db):
    """T99. Two overlapping submissions touching disjoint capability fields both land, and
    neither erases the other's columns — nor the channel's participation, placement or memory,
    which no capability save says anything about."""
    await capability_db.set_channel_settings_async(
        "C1", participation_level="on", reply_in_channel=True)
    await capability_db.add_channel_memory_async("C1", "deploys go out on thursdays")

    await asyncio.gather(
        capability_db.set_channel_settings_async("C1", model="gpt-5.5", enable_web_search=False),
        capability_db.set_channel_settings_async("C1", enable_mcp=True,
                                                 image_model="gpt-image-1"),
    )

    row = await capability_db.get_channel_settings_async("C1")
    assert (row["model"], row["enable_web_search"]) == ("gpt-5.5", 0)
    assert (row["enable_mcp"], row["image_model"]) == (1, "gpt-image-1")
    assert (row["participation_level"], row["reply_in_channel"]) == ("on", True)
    assert [m["content"] for m in await capability_db.get_channel_memory_async("C1")] == \
        ["deploys go out on thursdays"]


@pytest.mark.asyncio
async def test_an_old_modal_submission_preserves_the_new_fields(capability_db):
    """T101. A modal opened before W4 carries no capability blocks. Its Save must leave the
    channel's stored capabilities exactly as they are — an absent block is not a submission
    about that field, and clearing them would silently hand the channel back to the globals."""
    await capability_db.set_channel_settings_async(
        "C1", enable_web_search=False, enable_mcp=False, image_model="gpt-image-1")

    host = _settings_host(capability_db)
    old_state = {
        "channel_model_block": {"channel_model": {"selected_option": {"value": "gpt-5.5"}}},
        "channel_effort_block": {"channel_reasoning_effort": {"selected_option": {"value": "inherit"}}},
        "channel_verbosity_block": {"channel_verbosity": {"selected_option": {"value": "high"}}},
        "participation_block": {"participation_level": {"selected_option": {"value": "on"}}},
        "reply_in_channel_block": {"reply_in_channel": {"selected_option": {"value": "inherit"}}},
    }
    await host.app.views["channel_settings_modal"](
        ack=AsyncMock(),
        body={"user": {"id": "U1"}},
        view={"id": "V1", "private_metadata": json.dumps({"channel_id": "C1"}),
              "state": {"values": old_state}},
        client=AsyncMock(),
    )

    row = await capability_db.get_channel_settings_async("C1")
    assert (row["enable_web_search"], row["enable_mcp"]) == (0, 0)
    assert row["image_model"] == "gpt-image-1"
    # …while the fields that submission DID carry still landed.
    assert (row["model"], row["verbosity"]) == ("gpt-5.5", "high")


# The personal/DM view is pinned byte-for-byte (RULING-8), over a matrix that reaches both
# personal-modal branches the settings can steer: the welcome variant, and the reasoning=none
# variant that reveals the temperature / top-p inputs. `private_metadata` is dropped because it
# carries a fresh session uuid per open — a pointer to DB state, not part of the view.
#
# RE-CAPTURED for toolbelt T2, which adds the personal-memory section (textarea, "+N more"
# context, and the "forget everything" checkbox) directly under Custom Instructions. That is the
# ONLY intended change to this view; the digest is what makes a second, unintended one loud. The
# AsyncMock db yields no rows, so this pins the empty-store rendering.
_PERSONAL_MODAL_GOLDEN = "2f7d6b5c6427dcf11b3302d1a764f1e8850a423c56cb88a3cc2e95c6ce8f4775"


@pytest.mark.asyncio
async def test_the_personal_modal_is_byte_identical(monkeypatch):
    """T100."""
    from config import config as live_config

    monkeypatch.setattr(live_config, "settings_slash_command", "/settings-golden")
    monkeypatch.setattr(live_config, "default_verbosity", "medium")
    monkeypatch.setattr(live_config, "image_model", "gpt-image-2")
    modal = SettingsModal(db=AsyncMock())

    views = []
    for is_new_user in (False, True):
        for effort in ("none", "high"):
            view = await modal.build_settings_modal(
                "U1", "trigger-1",
                current_settings={"model": "gpt-5.6-terra", "reasoning_effort": effort,
                                  "verbosity": "low", "enable_web_search": True,
                                  "enable_mcp": False, "enable_streaming": True,
                                  "custom_instructions": "be brief",
                                  "image_model": "gpt-image-1", "image_size": "1024x1536",
                                  "image_quality": "high", "image_background": "transparent",
                                  "input_fidelity": "low", "vision_detail": "high",
                                  "temperature": 0.7, "top_p": 0.9},
                is_new_user=is_new_user, in_thread=True)
            view.pop("private_metadata")
            views.append(view)

    digest = hashlib.sha256(
        json.dumps(views, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert digest == _PERSONAL_MODAL_GOLDEN


@pytest.mark.asyncio
async def test_a_capability_submission_reaches_the_stored_row(capability_db):
    """The POSITIVE half of T101: choosing On / Off / an image model actually lands.

    T101 proves an absent block preserves, which a handler that extracted nothing at all would
    also satisfy — deleting the extraction, or the `**capability_kwargs` splat, escapes every
    other test in this wave. So this drives the real submission handler and reads the row back,
    including the round trip through 'inherit' that clears an override to NULL.
    """
    host = _settings_host(capability_db)

    async def submit(web_search, mcp, image_model):
        state = {
            "channel_model_block": {"channel_model": {"selected_option": {"value": "inherit"}}},
            "channel_effort_block": {"channel_reasoning_effort": {"selected_option": {"value": "inherit"}}},
            "channel_verbosity_block": {"channel_verbosity": {"selected_option": {"value": "inherit"}}},
            "participation_block": {"participation_level": {"selected_option": {"value": "on"}}},
            "reply_in_channel_block": {"reply_in_channel": {"selected_option": {"value": "inherit"}}},
            "channel_web_search_block": {"channel_enable_web_search": {"selected_option": {"value": web_search}}},
            "channel_mcp_block": {"channel_enable_mcp": {"selected_option": {"value": mcp}}},
            "channel_image_model_block": {"channel_image_model": {"selected_option": {"value": image_model}}},
        }
        await host.app.views["channel_settings_modal"](
            ack=AsyncMock(),
            body={"user": {"id": "U1"}},
            view={"id": "V1", "private_metadata": json.dumps({"channel_id": "C1"}),
                  "state": {"values": state}},
            client=AsyncMock())
        return await capability_db.get_channel_settings_async("C1")

    row = await submit("off", "on", "gpt-image-1")
    assert (row["enable_web_search"], row["enable_mcp"]) == (0, 1)
    assert row["image_model"] == "gpt-image-1"

    # 'inherit' on a control that IS on screen clears the override back to NULL — the opposite
    # statement from T101's absent block, and the two must not be confused.
    row = await submit("inherit", "inherit", "inherit")
    assert (row["enable_web_search"], row["enable_mcp"], row["image_model"]) == (None, None, None)


def test_an_unsaved_capability_choice_survives_the_model_rebuild():
    """The model select dispatches a re-render, and an in-flight edit has to survive it
    (SHOULD-FIX #7). Without the overlay carrying these three, changing the model silently
    reverts an unsaved web-search / MCP / image-model choice to whatever is stored.

    Driven end to end: form state → `_overlay_channel_form_state` → the rebuilt view, because
    the claim is about what the operator sees after the rebuild, not about a dict in between.
    """
    from slack_client.event_handlers.settings import SlackSettingsHandlersMixin

    stored = {"model": "gpt-5.6-sol", "enable_web_search": 1, "enable_mcp": 1,
              "image_model": "gpt-image-2"}
    state = {
        "channel_model_block": {"channel_model": {"selected_option": {"value": "gpt-5.5"}}},
        "channel_effort_block": {"channel_reasoning_effort": {"selected_option": {"value": "inherit"}}},
        "channel_verbosity_block": {"channel_verbosity": {"selected_option": {"value": "inherit"}}},
        "participation_block": {"participation_level": {"selected_option": {"value": "on"}}},
        "reply_in_channel_block": {"reply_in_channel": {"selected_option": {"value": "inherit"}}},
        # Three unsaved edits: web search switched off, MCP moved back to inherit, image model changed.
        "channel_web_search_block": {"channel_enable_web_search": {"selected_option": {"value": "off"}}},
        "channel_mcp_block": {"channel_enable_mcp": {"selected_option": {"value": "inherit"}}},
        "channel_image_model_block": {"channel_image_model": {"selected_option": {"value": "gpt-image-1"}}},
    }

    overlaid = SlackSettingsHandlersMixin._overlay_channel_form_state(stored, state)
    view = _capability_modal().build_channel_settings_modal("C1", overlaid, "tag_only")
    by_block = {b["block_id"]: b for b in view["blocks"] if b.get("block_id")}

    def selected(block_id):
        return by_block[block_id]["element"]["initial_option"]["value"]

    assert selected("channel_web_search_block") == "off"      # the edit, not the stored 1
    assert selected("channel_mcp_block") == "inherit"         # cleared, not resurrected as on
    assert selected("channel_image_model_block") == "gpt-image-1"


def test_the_personal_image_options_follow_the_shared_constant(monkeypatch):
    """A hard-coded option list is a latent view rejection: the coercion above validates against
    SUPPORTED_IMAGE_MODELS, so the moment that constant gains a model, a person holding it would
    get an `initial_option` absent from `options` and Slack refuses the whole view."""
    import config as config_module

    monkeypatch.setattr(config_module, "SUPPORTED_IMAGE_MODELS",
                        ("gpt-image-3", "gpt-image-2", "gpt-image-1"))
    blocks = SettingsModal.__new__(SettingsModal)._add_common_settings(
        {"image_model": "gpt-image-3"})
    accessory = next(b for b in blocks
                     if b.get("block_id") == "image_model_block")["accessory"]
    values = [o["value"] for o in accessory["options"]]
    assert values == ["gpt-image-3", "gpt-image-2", "gpt-image-1"]
    assert accessory["initial_option"]["value"] in values
