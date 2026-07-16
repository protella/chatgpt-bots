"""Channel settings modal — channel-memory editor + workspace-shared read-only list.

Channel-scope memory is now ONE editable multiline textarea (block_id "channel_memory_block",
action_id "channel_memory") that always renders — it doubles as the add surface. The seed of the
rows shown (``[[id, hash], ...]``) rides in ``private_metadata`` so submit can reconcile exactly
what the user saw. Workspace-scope memory stays read-only. Also covers the tri-state
reply_in_channel placement control. (The per-thread mute section and the per-row delete /
clear-mute block-action handlers were removed with the mute mechanism.)

Pure builder tests use a db-less SettingsModal. No live Slack, no API.
"""
from __future__ import annotations

import json

from database import memory_content_hash, normalize_memory_line
from settings_modal import SettingsModal


# --------------------------------------------------------------------------- helpers
def _builder() -> SettingsModal:
    # The builder needs no db (matches the other channel-modal tests).
    return SettingsModal.__new__(SettingsModal)


def _blocks(view) -> list:
    return view["blocks"]


def _memory_input(view) -> dict:
    """The channel_memory textarea element, or None."""
    for b in _blocks(view):
        if b.get("block_id") == "channel_memory_block":
            return b
    return None


def _all_text(view) -> str:
    return json.dumps(view["blocks"])


def _seed(view) -> list:
    return json.loads(view["private_metadata"]).get("mem_seed", [])


# --------------------------------------------------------------------------- memory textarea
class TestMemoryTextarea:
    def test_channel_scope_renders_single_multiline_input(self):
        mems = [
            {"id": 1, "scope": "channel", "content": "deploys break on fridays"},
            {"id": 2, "scope": "channel", "content": "prefer bullet points"},
        ]
        view = _builder().build_channel_settings_modal("C1", None, "tag_only", channel_memories=mems)
        block = _memory_input(view)
        assert block is not None
        assert block["type"] == "input" and block.get("optional") is True
        el = block["element"]
        assert el["type"] == "plain_text_input"
        assert el["action_id"] == "channel_memory"
        assert el["multiline"] is True
        assert el["max_length"] == 2900
        # both channel-scope notes seed the textarea, one per line
        assert el["initial_value"] == "deploys break on fridays\nprefer bullet points"
        assert "*What I remember about this channel*" in _all_text(view)

    def test_section_always_renders_even_with_zero_memories(self):
        # The section doubles as an add surface, so it renders even with nothing stored (unlike the
        # old per-row list which vanished when empty). No initial_value is set on an empty box.
        for arg in (None, []):
            view = _builder().build_channel_settings_modal("C1", None, "tag_only", channel_memories=arg)
            block = _memory_input(view)
            assert block is not None
            assert "initial_value" not in block["element"]
            assert "*What I remember about this channel*" in _all_text(view)
            assert _seed(view) == []

    def test_seed_lists_shown_channel_rows_with_hashes(self):
        mems = [
            {"id": 5, "scope": "channel", "content": "alpha"},
            {"id": 6, "scope": "channel", "content": "beta"},
            {"id": 7, "scope": "workspace", "content": "company holiday"},
        ]
        view = _builder().build_channel_settings_modal("C1", None, "tag_only", channel_memories=mems)
        # Only the channel-scope rows are seeded (workspace is read-only, never reconciled).
        assert _seed(view) == [[5, memory_content_hash("alpha")], [6, memory_content_hash("beta")]]

    def test_multiline_legacy_fact_collapses_to_one_line(self):
        # A legacy fact stored with embedded newlines must render as ONE textarea line (one note per
        # line is the contract), and its seed hash is the normalized-content hash.
        mems = [{"id": 9, "scope": "channel", "content": "deploys are\nThursday   mornings"}]
        view = _builder().build_channel_settings_modal("C1", None, "tag_only", channel_memories=mems)
        el = _memory_input(view)["element"]
        assert el["initial_value"] == "deploys are Thursday mornings"
        assert "\n" not in el["initial_value"]
        assert _seed(view) == [[9, memory_content_hash("deploys are\nThursday   mornings")]]

    def test_blank_channel_rows_dropped_from_textarea_and_seed(self):
        mems = [
            {"id": 1, "scope": "channel", "content": "   "},       # whitespace-only → dropped
            {"id": 2, "scope": "channel", "content": "real note"},
        ]
        view = _builder().build_channel_settings_modal("C1", None, "tag_only", channel_memories=mems)
        el = _memory_input(view)["element"]
        assert el["initial_value"] == "real note"
        assert _seed(view) == [[2, memory_content_hash("real note")]]

    def test_over_budget_rows_become_plus_n_more_and_are_not_seeded(self):
        # Rows past the 2900-char textarea budget are omitted from BOTH the box and the seed (never
        # seed a row the user can't see, or submit could "delete" it) and summarized as "+N more".
        big = "x" * 1500
        mems = [{"id": i, "scope": "channel", "content": f"{big}-{i}"} for i in range(3)]
        view = _builder().build_channel_settings_modal("C1", None, "tag_only", channel_memories=mems)
        seed = _seed(view)
        assert len(seed) < 3                                   # not all rows fit
        seeded_ids = {sid for sid, _ in seed}
        el = _memory_input(view)["element"]
        # every seeded row's content is actually in the box; the omitted ones are not
        for m in mems:
            content = normalize_memory_line(m["content"])
            assert (content in el["initial_value"]) == (m["id"] in seeded_ids)
        assert f"+{3 - len(seed)} more not shown" in _all_text(view)

    def test_workspace_scope_is_read_only_and_not_in_textarea(self):
        mems = [
            {"id": 1, "scope": "channel", "content": "channel note"},
            {"id": 2, "scope": "workspace", "content": "company holiday is july 4"},
        ]
        view = _builder().build_channel_settings_modal("C1", None, "tag_only", channel_memories=mems)
        text = _all_text(view)
        assert "company holiday is july 4" in text
        assert "read-only" in text.lower()
        el = _memory_input(view)["element"]
        assert "company holiday is july 4" not in el["initial_value"]  # never editable here


# --------------------------------------------------------------------------- reply_in_channel tri-state (SHOULD-FIX #5)
class TestReplyInChannelTriState:
    """The placement control is a tri-state static_select: a stored NULL pre-selects 'inherit'
    (never resolved to today's global default — that's what froze the value on save before), an
    explicit True → 'channel', an explicit False → 'threads'."""

    def _reply_element(self, view):
        return next(b for b in _blocks(view)
                    if b.get("block_id") == "reply_in_channel_block")["element"]

    def _selected(self, view):
        el = self._reply_element(view)
        assert el["type"] == "static_select"
        return el["initial_option"]["value"]

    def test_null_shows_inherit_regardless_of_default(self, monkeypatch):
        from config import config
        # Stored NULL must ALWAYS pre-select 'inherit', whatever the global default is — otherwise
        # opening + saving untouched would freeze that default into an explicit row.
        for default in (True, False):
            monkeypatch.setattr(config, "reply_in_channel_default", default)
            view = _builder().build_channel_settings_modal("C1", {"reply_in_channel": None}, "tag_only")
            assert self._selected(view) == "inherit"

    def test_absent_key_shows_inherit(self):
        view = _builder().build_channel_settings_modal("C1", {}, "tag_only")
        assert self._selected(view) == "inherit"

    def test_explicit_true_shows_channel(self):
        view = _builder().build_channel_settings_modal("C1", {"reply_in_channel": True}, "tag_only")
        assert self._selected(view) == "channel"

    def test_explicit_false_shows_threads(self):
        view = _builder().build_channel_settings_modal("C1", {"reply_in_channel": False}, "tag_only")
        assert self._selected(view) == "threads"

    def test_inherit_option_notes_effective_default(self, monkeypatch):
        from config import config
        monkeypatch.setattr(config, "reply_in_channel_default", True)
        view = _builder().build_channel_settings_modal("C1", {"reply_in_channel": None}, "tag_only")
        inherit = next(o for o in self._reply_element(view)["options"] if o["value"] == "inherit")
        assert "channel level" in inherit["text"]["text"].lower()

        monkeypatch.setattr(config, "reply_in_channel_default", False)
        view = _builder().build_channel_settings_modal("C1", {"reply_in_channel": None}, "tag_only")
        inherit = next(o for o in self._reply_element(view)["options"] if o["value"] == "inherit")
        assert "threads only" in inherit["text"]["text"].lower()
