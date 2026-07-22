"""Participation-backoff redesign (Layer 2) — expanded verdict + _apply_backoff taxonomy.

Covers:
- verdict parsing: backward-compat (old {action,emoji,placement,reason} still parses) and the
  new backoff taxonomy fields (dimension/durability/scope/guidance/memory_op/structural_request),
  including malformed-field degradation and the optional backoff ack emoji.
- each _apply_backoff branch: standing CHANNEL-scope soft pref → per-channel/per-dimension memory
  (add / marker dedup / cap) and NEVER a settings write; momentary → nothing durable; thread-scope
  (any durability) → nothing durable (the per-thread mute mechanism was removed); channel reversal
  → delete pref memory; explicit structural request → falls through (returns True) with no write.
- the conditional ack: routed through the reservation path; never when dimension == reactions.

The classifier and DB are mocked; no real API/DB calls.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message
from config import config
from message_processor.participation import ParticipationEngine, ParticipationVerdict


# ----------------------------------------------------------------- verdict parsing

class TestExpandedVerdictParsing:
    def test_old_shape_still_parses_with_defaults(self):
        v = ParticipationEngine.validate_verdict(
            {"action": "respond", "emoji": None, "placement": "channel", "reason": "hi"})
        assert (v.action, v.placement) == ("respond", "channel")
        # new fields default and never leak onto a non-backoff verdict
        assert v.dimension is None and v.durability is None and v.scope is None
        assert v.guidance == "" and v.memory_op == "none" and v.structural_request == "none"

    def test_backoff_taxonomy_parsed(self):
        v = ParticipationEngine.validate_verdict({
            "action": "backoff", "dimension": "reactions", "durability": "standing",
            "scope": "channel", "guidance": "React less here", "memory_op": "add",
            "structural_request": "none",
        })
        assert v.dimension == "reactions" and v.durability == "standing"
        assert v.scope == "channel" and v.guidance == "React less here"
        assert v.memory_op == "add" and v.structural_request == "none"

    def test_backoff_malformed_fields_degrade_safe(self):
        v = ParticipationEngine.validate_verdict({
            "action": "backoff", "dimension": "loudness", "durability": "forever",
            "scope": "planet", "memory_op": "obliterate", "structural_request": "everything",
        })
        assert v.dimension is None and v.durability is None and v.scope is None
        assert v.memory_op == "none" and v.structural_request == "none"

    def test_memory_op_variants(self):
        def mk(op):
            return ParticipationEngine.validate_verdict(
                {"action": "backoff", "memory_op": op}).memory_op
        assert mk("delete") == "delete"
        assert mk("delete:12") == "delete:12"
        assert mk("update:7") == "update:7"
        assert mk("update") == "none"       # update needs a target id
        assert mk("add") == "add"

    def test_backoff_optional_ack_emoji(self, monkeypatch):
        monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
        assert ParticipationEngine.validate_verdict(
            {"action": "backoff", "emoji": ":thumbsup:"}).emoji == "thumbsup"
        # no/garbage emoji on a backoff simply means no ack (never forced)
        assert ParticipationEngine.validate_verdict({"action": "backoff"}).emoji is None

    def test_non_backoff_never_carries_taxonomy(self):
        v = ParticipationEngine.validate_verdict(
            {"action": "react", "emoji": "eyes", "dimension": "reactions", "memory_op": "add"})
        assert v.memory_op == "none" and v.dimension is None


# ----------------------------------------------------------------- _apply_backoff branches

def _app():
    from main import ChatBotV2
    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    db = MagicMock()
    db.get_channel_memory_async = AsyncMock(return_value=[])
    db.add_channel_memory_async = AsyncMock(return_value=99)
    db.update_channel_memory_async = AsyncMock()
    db.delete_channel_memory_async = AsyncMock()
    # The add/refresh path now goes through the atomic pref upsert (SHOULD-FIX #8); it returns the
    # marker row id (or None when declined at cap — overridden per-test).
    db.upsert_channel_pref_memory = AsyncMock(return_value=99)
    db.set_channel_settings_async = AsyncMock()
    app.processor.db = db
    client = MagicMock()
    client.react = AsyncMock()
    client._reserve_and_react = AsyncMock(return_value={"ok": True})
    return app, client, db


def _msg(**meta):
    m = {"ts": "50.0"}
    m.update(meta)
    return Message(text="feedback", user_id="U1", channel_id="C1", thread_id="50.0", metadata=m)


def _verdict(**kw):
    return ParticipationVerdict(action="backoff", **kw)


@pytest.mark.asyncio
async def test_standing_soft_pref_writes_memory_only(monkeypatch):
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    app, client, db = _app()
    v = _verdict(dimension="reactions", durability="standing", scope="channel",
                 guidance="React less in this channel", memory_op="add")
    fell_through = await app._apply_backoff(_msg(), client, v)
    assert fell_through is False
    # ONE per-channel/per-dimension pref memory, written atomically via the marker upsert.
    call = db.upsert_channel_pref_memory.await_args
    assert call.args[0] == "C1"                                    # channel_id
    assert call.args[1] == "participation_engine:pref:reactions"   # stable marker author
    assert "React less" in call.args[2]                           # content
    assert call.kwargs["max_rows"] == config.memory_max_rows       # cap flows to the helper
    # never the raw add (which can't enforce the single-row invariant), never a structural
    # settings write (the clobber the redesign fixes)
    db.add_channel_memory_async.assert_not_awaited()
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_standing_pref_repeat_routes_through_single_row_upsert(monkeypatch):
    # A repeat "react less" refreshes the ONE marker row rather than accumulating duplicates. At
    # the engine level that means routing through the atomic upsert (which collapses to one row —
    # the real dedup is pinned at the DB level in test_channel_thread_mutes.py), never the raw
    # add/update that can't hold the single-row invariant.
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    app, client, db = _app()
    v = _verdict(dimension="reactions", durability="standing", scope="channel",
                 guidance="React even less", memory_op="add")
    await app._apply_backoff(_msg(), client, v)
    db.upsert_channel_pref_memory.assert_awaited_once()
    args = db.upsert_channel_pref_memory.await_args
    assert args.args[1] == "participation_engine:pref:reactions"
    assert "React even less" in args.args[2]
    db.add_channel_memory_async.assert_not_awaited()
    db.update_channel_memory_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_standing_pref_at_cap_declined_by_helper_is_handled(monkeypatch):
    # The MEMORY_MAX_ROWS cap now lives INSIDE the atomic helper (it alone can enforce it without
    # a read-then-insert race). The engine passes the cap through and handles a decline (helper
    # returns None) without crashing or falling back to a raw add.
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    monkeypatch.setattr(config, "memory_max_rows", 2, raising=False)
    app, client, db = _app()
    db.upsert_channel_pref_memory = AsyncMock(return_value=None)  # helper declined at cap
    v = _verdict(dimension="verbosity", durability="standing", scope="channel", memory_op="add")
    await app._apply_backoff(_msg(), client, v)
    db.upsert_channel_pref_memory.assert_awaited_once()
    assert db.upsert_channel_pref_memory.await_args.kwargs["max_rows"] == 2
    db.add_channel_memory_async.assert_not_awaited()
    db.update_channel_memory_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_momentary_persists_nothing():
    app, client, db = _app()
    v = _verdict(dimension="replies", durability="momentary", scope="channel", memory_op="none")
    fell_through = await app._apply_backoff(_msg(), client, v)
    assert fell_through is False
    db.add_channel_memory_async.assert_not_awaited()
    db.upsert_channel_pref_memory.assert_not_awaited()
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_thread_scope_standing_persists_nothing():
    # The per-thread mute mechanism was removed: a standing, THREAD-scoped "stop replying here" is
    # guidance for the current message only — it writes nothing durable (no mute, no memory, no
    # settings), exactly like a momentary aside.
    app, client, db = _app()
    v = _verdict(dimension="thread_participation", durability="standing", scope="thread",
                 guidance="stay out of this thread", memory_op="add")
    fell_through = await app._apply_backoff(_msg(ts="50.0"), client, v)
    assert fell_through is False
    db.upsert_channel_pref_memory.assert_not_awaited()
    db.add_channel_memory_async.assert_not_awaited()
    db.delete_channel_memory_async.assert_not_awaited()
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_thread_scope_reversal_persists_nothing():
    # A thread-scoped reversal ("you can chime back in here") likewise has no durable store to
    # undo now — it must not attempt any channel-memory delete or settings write.
    app, client, db = _app()
    v = _verdict(durability="standing", scope="thread", memory_op="delete")
    fell_through = await app._apply_backoff(_msg(ts="50.0"), client, v)
    assert fell_through is False
    db.delete_channel_memory_async.assert_not_awaited()
    db.upsert_channel_pref_memory.assert_not_awaited()
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_channel_reversal_deletes_pref_memory(monkeypatch):
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    app, client, db = _app()
    db.get_channel_memory_async = AsyncMock(return_value=[
        {"id": 8, "author": "participation_engine:pref:reactions", "content": "react less", "scope": "channel"},
    ])
    v = _verdict(dimension="reactions", durability="standing", scope="channel", memory_op="delete:8")
    await app._apply_backoff(_msg(), client, v)
    db.delete_channel_memory_async.assert_awaited_once_with(8)


@pytest.mark.asyncio
async def test_structural_request_falls_through_without_writes():
    app, client, db = _app()
    v = _verdict(durability="standing", scope="channel", structural_request="participation")
    fell_through = await app._apply_backoff(_msg(), client, v)
    assert fell_through is True
    # nothing durable is written here — the model owns the settings change via the tool
    db.add_channel_memory_async.assert_not_awaited()
    db.upsert_channel_pref_memory.assert_not_awaited()
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_structural_request_stamps_gate_authorized_flag():
    # BLOCKER #3: routing a structural request into the response loop STAMPS the turn
    # (gate_authorized_structural) so the gated set_channel_participation tool is authorized there
    # — even with no literal <@bot> mention ("only reply when I tag you" carries none). The
    # classifier's semantic judgment, not the raw name regex, is what authorizes.
    app, client, db = _app()
    msg = _msg()
    v = _verdict(durability="standing", scope="channel", structural_request="placement")
    assert await app._apply_backoff(msg, client, v) is True
    assert msg.metadata.get("gate_authorized_structural") is True


@pytest.mark.asyncio
async def test_non_structural_backoff_does_not_stamp_gate_flag(monkeypatch):
    # A non-structural backoff (a soft per-channel preference) is NOT a settings request and must
    # NOT stamp the authorization flag — otherwise "you're a bit chatty" would authorize the tool.
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    app, client, db = _app()
    msg = _msg()
    v = _verdict(dimension="reactions", durability="standing", scope="channel", memory_op="add")
    await app._apply_backoff(msg, client, v)
    assert msg.metadata.get("gate_authorized_structural") is not True


# ----------------------------------------------------------------- conditional ack

@pytest.mark.asyncio
async def test_ack_emoji_routed_through_reservation():
    app, client, db = _app()
    v = _verdict(dimension="replies", durability="momentary", scope="channel", emoji="thumbsup")
    await app._apply_backoff(_msg(ts="50.0"), client, v)
    client._reserve_and_react.assert_awaited_once_with("C1", "50.0", "thumbsup")


@pytest.mark.asyncio
async def test_never_acks_when_dimension_is_reactions():
    app, client, db = _app()
    v = _verdict(dimension="reactions", durability="standing", scope="channel",
                 memory_op="add", emoji="thumbsup")
    await app._apply_backoff(_msg(), client, v)
    client._reserve_and_react.assert_not_awaited()
    client.react.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_ack_when_no_emoji():
    app, client, db = _app()
    v = _verdict(dimension="replies", durability="momentary", scope="channel", emoji=None)
    await app._apply_backoff(_msg(), client, v)
    client._reserve_and_react.assert_not_awaited()


# ----------------------------------------------------------------- adversarial cases
# The scenarios the redesign spec calls out (soft reaction feedback, negation, quoted/third-party
# speech, joking/teasing, momentary "not now", explicit thread-exit/reversal/settings) that the
# happy-path branch tests above don't already pin down.

@pytest.mark.asyncio
async def test_channel_negation_bare_delete_removes_marker_pref(monkeypatch):
    # Negation ("don't stop reacting" / "you can react again") that the classifier resolves to a
    # channel reversal WITHOUT a specific [#id] — a BARE `delete` — must remove the marker
    # preference row (matched by its stable author), un-suppressing the assistant. It must never
    # be mistaken for a fresh suppression: no add, no mute, no settings write.
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    app, client, db = _app()
    db.get_channel_memory_async = AsyncMock(return_value=[
        {"id": 8, "author": "participation_engine:pref:reactions",
         "content": "react less here", "scope": "channel"},
    ])
    v = _verdict(dimension="reactions", durability="standing", scope="channel", memory_op="delete")
    fell_through = await app._apply_backoff(_msg(), client, v)
    assert fell_through is False
    db.delete_channel_memory_async.assert_awaited_once_with(8)
    db.add_channel_memory_async.assert_not_awaited()
    db.set_channel_settings_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_quoted_third_party_feedback_ignored_persists_nothing():
    # "Bob told you to pipe down" reported by someone else is NOT a real instruction. The
    # classifier resolves quoted/third-party speech to `ignore` (never `backoff`), so it never
    # reaches _apply_backoff at all — but if a malformed verdict smuggled `backoff` through with
    # no taxonomy (durability/scope/structural all defaulted), the router still writes nothing.
    app, client, db = _app()
    v = _verdict()  # bare backoff: durability=None, scope=None, structural_request="none"
    fell_through = await app._apply_backoff(_msg(), client, v)
    assert fell_through is False
    db.add_channel_memory_async.assert_not_awaited()
    db.update_channel_memory_async.assert_not_awaited()
    db.upsert_channel_pref_memory.assert_not_awaited()
    db.set_channel_settings_async.assert_not_awaited()
    client._reserve_and_react.assert_not_awaited()


@pytest.mark.asyncio
async def test_thread_scope_standing_persists_nothing_any_dimension(monkeypatch):
    # THE CLOBBER REGRESSION at the engine level, now that mutes are gone: a standing, thread-scoped
    # backoff for EVERY feedback dimension writes NOTHING durable — no channel memory, no marker
    # upsert, and above all no structural settings (response_mode / reply_in_channel / level).
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    for dimension in ("reactions", "replies", "verbosity", "thread_participation"):
        app, client, db = _app()
        v = _verdict(dimension=dimension, durability="standing", scope="thread", memory_op="add")
        await app._apply_backoff(_msg(ts="50.0"), client, v)
        db.set_channel_settings_async.assert_not_awaited()
        db.add_channel_memory_async.assert_not_awaited()
        db.upsert_channel_pref_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_momentary_reactions_feedback_writes_nothing_and_never_acks():
    # A joking / in-the-moment "ok ok, stop reacting to everything 😄" — momentary, about
    # reactions — persists nothing (it should be forgotten immediately) AND never acks with a
    # reaction (acking "stop reacting" with a reaction is exactly wrong), even if an emoji leaked.
    app, client, db = _app()
    v = _verdict(dimension="reactions", durability="momentary", scope="channel", emoji="thumbsup")
    fell_through = await app._apply_backoff(_msg(), client, v)
    assert fell_through is False
    db.add_channel_memory_async.assert_not_awaited()
    db.upsert_channel_pref_memory.assert_not_awaited()
    db.set_channel_settings_async.assert_not_awaited()
    client._reserve_and_react.assert_not_awaited()
    client.react.assert_not_awaited()


# ------------------------------------------------- BLOCKER #4: backoff CRUD scope discipline
# The engine may CRUD only its OWN per-dimension preference markers. A verdict id that names a
# workspace fact or a human's channel fact must NEVER be updated or deleted — it falls back to the
# safe marker path instead. (An injected/hallucinated `update:<id>`/`delete:<id>` is the vector.)

@pytest.mark.asyncio
async def test_update_targeting_workspace_fact_is_refused_and_upserts_marker(monkeypatch):
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    app, client, db = _app()
    db.get_channel_memory_async = AsyncMock(return_value=[
        {"id": 3, "author": "admin", "content": "workspace policy", "scope": "workspace"},
    ])
    v = _verdict(dimension="reactions", durability="standing", scope="channel",
                 guidance="react less", memory_op="update:3")
    await app._apply_backoff(_msg(), client, v)
    # the workspace row is NEVER rewritten or deleted…
    db.update_channel_memory_async.assert_not_awaited()
    db.delete_channel_memory_async.assert_not_awaited()
    # …instead the write falls back to the engine's own per-dimension marker upsert
    db.upsert_channel_pref_memory.assert_awaited_once()
    assert db.upsert_channel_pref_memory.await_args.args[1] == "participation_engine:pref:reactions"


@pytest.mark.asyncio
async def test_delete_targeting_human_channel_fact_is_refused(monkeypatch):
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    app, client, db = _app()
    db.get_channel_memory_async = AsyncMock(return_value=[
        {"id": 7, "author": "U42", "content": "we deploy on fridays", "scope": "channel"},
    ])
    v = _verdict(dimension="reactions", durability="standing", scope="channel", memory_op="delete:7")
    await app._apply_backoff(_msg(), client, v)
    # a human's channel fact is never deleted; with no owned marker to fall back to, it no-ops
    db.delete_channel_memory_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_bad_id_falls_back_to_own_marker_not_the_human_fact(monkeypatch):
    # `delete:7` names a human fact (refused) but an owned marker for this dimension also exists →
    # the reversal removes the MARKER (id 8), never the human fact (id 7).
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    app, client, db = _app()
    db.get_channel_memory_async = AsyncMock(return_value=[
        {"id": 7, "author": "U42", "content": "we deploy on fridays", "scope": "channel"},
        {"id": 8, "author": "participation_engine:pref:reactions",
         "content": "react less here", "scope": "channel"},
    ])
    v = _verdict(dimension="reactions", durability="standing", scope="channel", memory_op="delete:7")
    await app._apply_backoff(_msg(), client, v)
    db.delete_channel_memory_async.assert_awaited_once_with(8)


# ------------------------------------------------- SHOULD-FIX 1: cross-dimension marker discipline
# A verdict for one dimension must never corrupt a DIFFERENT dimension's preference. `_own_pref_row`
# alone matched ANY engine marker; the fix requires the target row's author to equal THIS
# dimension's marker, so a `reactions` verdict can't rewrite/delete the `verbosity` row.

@pytest.mark.asyncio
async def test_update_targeting_other_dimension_marker_is_refused(monkeypatch):
    # A `reactions` verdict with update:<id> pointing at the `verbosity` marker must NOT rewrite it
    # — it falls back to the reactions marker's own upsert, leaving verbosity untouched.
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    app, client, db = _app()
    db.get_channel_memory_async = AsyncMock(return_value=[
        {"id": 5, "author": "participation_engine:pref:verbosity",
         "content": "be brief", "scope": "channel"},
    ])
    v = _verdict(dimension="reactions", durability="standing", scope="channel",
                 guidance="react less", memory_op="update:5")
    await app._apply_backoff(_msg(), client, v)
    db.update_channel_memory_async.assert_not_awaited()   # the verbosity row is never rewritten…
    db.upsert_channel_pref_memory.assert_awaited_once()   # …the write lands on the reactions marker
    assert db.upsert_channel_pref_memory.await_args.args[1] == "participation_engine:pref:reactions"


@pytest.mark.asyncio
async def test_delete_targeting_other_dimension_marker_is_refused(monkeypatch):
    # A delete:<id> naming ANOTHER dimension's marker must NOT delete it. With no marker for THIS
    # dimension to fall back to, the reversal no-ops rather than deleting the verbosity preference.
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    app, client, db = _app()
    db.get_channel_memory_async = AsyncMock(return_value=[
        {"id": 5, "author": "participation_engine:pref:verbosity",
         "content": "be brief", "scope": "channel"},
    ])
    v = _verdict(dimension="reactions", durability="standing", scope="channel", memory_op="delete:5")
    await app._apply_backoff(_msg(), client, v)
    db.delete_channel_memory_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_other_dimension_id_falls_back_to_own_marker(monkeypatch):
    # delete:5 names the verbosity marker (refused) but a reactions marker (id 9) also exists →
    # the reversal removes the REACTIONS marker (this dimension's own row), never verbosity.
    monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
    app, client, db = _app()
    db.get_channel_memory_async = AsyncMock(return_value=[
        {"id": 5, "author": "participation_engine:pref:verbosity",
         "content": "be brief", "scope": "channel"},
        {"id": 9, "author": "participation_engine:pref:reactions",
         "content": "react less", "scope": "channel"},
    ])
    v = _verdict(dimension="reactions", durability="standing", scope="channel", memory_op="delete:5")
    await app._apply_backoff(_msg(), client, v)
    db.delete_channel_memory_async.assert_awaited_once_with(9)


# ----------------------------------------------------------------- classifier prompt guardrails
# The taxonomy is only as safe as the prompt that fills it in. These pin the incident-critical
# instructions: the classifier must derive durable changes ONLY from an explicit current-message
# instruction — never from memory, history, quoted/third-party speech, jokes, or image text.

class TestParticipationPromptGuardrails:
    def test_prompt_documents_backoff_taxonomy_fields(self):
        from prompts import PARTICIPATION_SYSTEM_PROMPT as p
        for field in ("dimension", "durability", "scope", "guidance",
                      "memory_op", "structural_request"):
            assert field in p
        for enum in ("reactions", "replies", "verbosity", "thread_participation",
                     "momentary", "standing"):
            assert enum in p

    def test_prompt_durable_changes_only_from_explicit_current_message(self):
        # The core defense against the incident and against quoted/third-party feedback: nothing
        # durable may be inferred from memory, earlier history, quoted/reported speech, or image
        # text — only an explicit instruction in the CURRENT message.
        from prompts import PARTICIPATION_SYSTEM_PROMPT as p
        assert "Only an explicit, direct instruction in the CURRENT message changes anything durable" in p
        for src in ("channel memory", "earlier history", "quoted or reported speech", "image"):
            assert src in p

    def test_prompt_structural_request_is_explicit_only_not_soft_feedback(self):
        # The exact incident: a vague "you're a bit chatty" must be a remembered preference, NOT a
        # structural settings change. structural_request is reserved for an explicit instruction.
        from prompts import PARTICIPATION_SYSTEM_PROMPT as p
        assert "NOT a structural request" in p
        assert "explicit, direct instruction to change the CHANNEL'S settings" in p

    def test_prompt_reversal_is_a_backoff_case(self):
        # Negation / reversal ("you can chime in again", "react away") is handled as backoff so it
        # can un-do a recorded suppression, not be mistaken for a new one.
        from prompts import PARTICIPATION_SYSTEM_PROMPT as p
        assert "you can chime in again" in p and "react away" in p

    def test_prompt_teasing_at_assistant_is_respond_not_backoff(self):
        # Item B: banter genuinely AT the assistant is participation-worthy (react or a quip),
        # not backoff; teasing pointed at another party still stays theirs.
        from prompts import PARTICIPATION_SYSTEM_PROMPT as p
        assert "teasing genuinely aimed AT the assistant is participation-worthy" in p
        assert "teasing pointed at another party, or a message merely talking about the assistant stays theirs" in p

    def test_prompt_forbids_reaction_ack_on_reactions_dimension(self):
        from prompts import PARTICIPATION_SYSTEM_PROMPT as p
        assert 'NEVER set it when "dimension" is "reactions"' in p
