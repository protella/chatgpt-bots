"""Grammar v3 of the channel stream, frozen byte for byte.

This is the contract the prompt cache, the stale-send guard and every cross-thread awareness
claim rest on. `serialize_stream` is a pure function of the pinned tuple, so the tests are
literal expected strings rather than shape assertions: if a header, a marker or an escape moves,
that is a serializer version change and it has to be a deliberate one.

The MESSAGE-item grammar below is v1's, unchanged, and it stays asserted here: v3 changed which
items exist and what the framing items say, not how a message renders. What v3 removed — the
summary item and the anchor map — is asserted absent here and tripwired in
`test_retired_machinery.py`; the escaping and the freeze it kept are asserted below.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from message_processor import channel_stream
from message_processor.channel_stream import (
    A2_RESERVED_PREFIXES,
    DELETED_SNIPPET,
    END_MARKER_TEXT,
    HORIZON_TEMPLATE,
    INVENTORY_STATES,
    SERIALIZER_VERSION,
    ChannelStream,
    FreezeError,
    InventoryPin,
    PinnedTuple,
    ReceiptRec,
    SidecarPin,
    StreamTimestampError,
    _dedup,
    _build_actor_map,
    _root_snippet,
    build_origin_pin,
    SharedPageCounts,
    SharedChannelPin,
    OriginFetch,
    classify_chrome,
    escape_payload,
    escape_payload_line,
    freeze_deep,
    index_clause,
    iso_minute,
    serialize_stream,
    serializer_config_snapshot,
)
from slack_client.utilities import (ACTOR_REMOTE_LOOKUP_DEFAULT,
                                    SlackUtilitiesMixin)
from slack_client.normalizer import (
    ORIGIN_HISTORY,
    ORIGIN_REPLIES,
    FileRef,
    NormalizedMessage,
    ReactionRec,
)

TEAM = "T1"
CH = "C0BKX77NU66"
# 1700000000 == 2023-11-14 22:13 UTC. Every stamp below is derived from that.
T0 = "1700000000.000100"
T1 = "1700000060.000200"
T2 = "1700003600.000300"
STAMP0 = "2023-11-14 22:13"
STAMP1 = "2023-11-14 22:14"
STAMP2 = "2023-11-14 23:13"
COVERAGE = InventoryPin(start_ts="1699999940.000000", status="complete", reason="genesis")
HORIZON = HORIZON_TEMPLATE.format(floor_ts=COVERAGE.start_ts, reach_clause="",
                                  index_clause="")


def msg(ts, *, text="hello", sender="U1", sender_type="human", root=None, files=(),
        reactions=(), edited_ts=None, broadcast=False, tombstone=False, reply_count=None,
        latest_reply=None, mentions=(), origin=ORIGIN_HISTORY, bot_name=None,
        subtype=None) -> NormalizedMessage:
    return NormalizedMessage(
        team_id=TEAM, channel_id=CH, ts=ts, thread_root_ts=root, subtype=subtype,
        sender_id=sender, sender_type=sender_type, raw_bot_name=bot_name, text=text,
        files=tuple(files), reactions=tuple(reactions), edited_ts=edited_ts,
        is_broadcast=broadcast, is_tombstone=tombstone, reply_count=reply_count,
        latest_reply=latest_reply, mention_ids=tuple(mentions), origin=origin)


def sidecars(*, receipts=(), epoch=None, images=(), documents=(), ambient=(),
             tools=()) -> SidecarPin:
    return SidecarPin(
        window=(COVERAGE.start_ts, True), receipts=tuple(receipts),
        receipt_feature_epoch_ts=epoch, coverage=COVERAGE, activity_roots=(),
        activity_event_ts=(), image_analyses=tuple(images),
        document_extractions=tuple(documents), ambient_artifacts=tuple(ambient),
        tool_usage=tuple(tools), versions_hash="sidecarhash")


def pinned(messages, *, cards=None, actors=(("U1", "alice"),), h=T2,
           serializer_config=None) -> PinnedTuple:
    cards = cards if cards is not None else sidecars()
    floor = cards.window[0]
    return PinnedTuple(
        team_id=TEAM, channel_id=CH, window=(floor, True), H=h,
        periphery_floor_ts=floor, selection_version=1,
        chrome_ts=classify_chrome(
            tuple(messages), chrome_markers=(serializer_config
                                             or serializer_config_snapshot())["chrome_markers"]),
        fetch_snapshot=tuple(messages), sidecar_versions_hash=cards.versions_hash,
        actor_map=tuple(actors), actor_map_hash="actorhash",
        serializer_version=SERIALIZER_VERSION, serializer_config_hash="cfghash",
        capability_profile_hash="caphash", tool_schema_version="tools-1",
        coverage=COVERAGE, receipt_feature_epoch_ts=cards.receipt_feature_epoch_ts,
        receipt_map=tuple((r.ts, r.state, r.turn_id, r.thread_root_ts) for r in cards.receipts),
        sidecars=cards,
        serializer_config=serializer_config or serializer_config_snapshot())


def contents(stream: ChannelStream):
    return [item.content for item in stream.items]


# ------------------------------------------------------------------ framing

def test_item_order_is_horizon_then_messages_then_end_marker():
    stream = serialize_stream(pinned([msg(T0), msg(T1)]))
    assert contents(stream)[0] == HORIZON
    assert contents(stream)[-1] == END_MARKER_TEXT
    assert stream.message_count == 2
    assert [i.role for i in stream.items] == ["user"] * 4


def test_horizon_carries_no_H():
    """The horizon is item 0 of the cacheable prefix, so it carries only slow-moving facts. A
    per-turn H here would change the first item every turn and invalidate the whole stream
    beneath it; the live edge is stated post-breakpoint instead."""
    stream = serialize_stream(pinned([msg(T0)], h=T2))
    assert T2 not in stream.horizon_item.content
    assert COVERAGE.start_ts in stream.horizon_item.content


def test_a_warm_index_adds_no_clause_and_a_limited_one_falls_back_to_retention():
    """§2f's fail-safe: a `limited` row with NO reason reads as Slack's retention, never as our
    configured depth. Claiming the depth cap would assert a cause we cannot prove."""
    assert serialize_stream(pinned([msg(T0)])).horizon_item.content == HORIZON
    pin = pinned([msg(T0)])
    object.__setattr__(pin, "coverage", InventoryPin(COVERAGE.start_ts, "limited", None))
    assert "Slack's retention limits how far back my thread index reaches" in (
        serialize_stream(pin).horizon_item.content)


def test_framing_items_carry_no_metadata_so_the_stale_guard_ignores_them():
    stream = serialize_stream(pinned([msg(T0)]))
    assert stream.horizon_item.metadata == {}
    assert stream.end_marker_item.metadata == {}


# ------------------------------------------------------------------ header grammar

def test_top_level_header_bytes():
    stream = serialize_stream(pinned([msg(T0, text="hi")]))
    assert contents(stream)[1] == (
        f"[{STAMP0} alice (human) id=U1 ts={T0}]\nhi")


def test_reply_header_carries_thread_and_root_snippet():
    root = msg(T0, text="the root message text that runs on and on past the cut")
    reply = msg(T1, text="ack", root=T0, origin=ORIGIN_REPLIES)
    stream = serialize_stream(pinned([root, reply]))
    assert contents(stream)[2] == (
        f'[{STAMP1} alice (human) id=U1 ts={T1} thread={T0}'
        f'~"the root message text that runs"]\nack')


def test_snippet_is_six_words_hard_cut_at_48_chars():
    root = msg(T0, text="alpha beta gamma delta epsilon zeta eta theta")
    reply = msg(T1, text="x", root=T0)
    header = contents(serialize_stream(pinned([root, reply])))[2].split("\n")[0]
    assert '~"alpha beta gamma delta epsilon zeta"' in header
    long_root = msg(T0, text="aaaaaaaaaa bbbbbbbbbb cccccccccc dddddddddd eeeeeeeeee ffff")
    header = contents(serialize_stream(pinned([long_root, reply])))[2].split("\n")[0]
    snippet = header.split('~"', 1)[1].rstrip('"]')
    assert len(snippet) == 48


def test_tombstoned_root_beats_the_snippet_branch():
    root = msg(T0, text="This message was deleted.", tombstone=True)
    reply = msg(T1, text="still here", root=T0)
    header = contents(serialize_stream(pinned([root, reply])))[2].split("\n")[0]
    assert header.endswith(f"thread={T0}{DELETED_SNIPPET}]")


def test_absent_root_renders_no_snippet():
    """A reply whose root is not in the window renders a ROOTLESS header. Under v3 it is also
    preceded by its own orphan-marker item, so the message is items[2] — the marker is what
    identifies the thread now, and the snippet would have leaked text the pin cannot vouch for."""
    reply = msg(T1, text="orphan", root="1699000000.000000")
    items = contents(serialize_stream(pinned([reply])))
    assert items[1].startswith("[thread=1699000000.000000 began before this window")
    header = items[2].split("\n")[0]
    assert header.endswith("thread=1699000000.000000]")


def test_flags_render_in_pinned_order():
    m = msg(T1, text="x", root=T0, edited_ts="1700000070.000000", broadcast=True)
    # Its root is absent from this one-message window, so an orphan marker precedes it.
    header = contents(serialize_stream(pinned([m])))[2].split("\n")[0]
    assert header.endswith(" (edited) (broadcast)]")


def test_name_falls_back_to_sender_id_then_unknown():
    stream = serialize_stream(pinned([msg(T0, sender="U9")], actors=()))
    assert contents(stream)[1].startswith(f"[{STAMP0} U9 (human) id=U9 ")
    stream = serialize_stream(pinned([msg(T0, sender=None)], actors=()))
    assert contents(stream)[1].startswith(f"[{STAMP0} unknown (human) id=unknown ")


def test_iso_minute_uses_the_seconds_field_not_a_float_round():
    # 1700000039 is 22:13:59; int(float("1700000039.9999999")) rounds UP to 1700000040 and
    # renders the next minute. The seconds field does not.
    assert int(float("1700000039.9999999")) == 1700000040
    assert iso_minute("1700000039.9999999") == STAMP0
    assert iso_minute("1700000040.000000") == STAMP1


def test_bot_sender_type_and_name_render():
    stream = serialize_stream(pinned(
        [msg(T0, sender="B7", sender_type="other_bot", bot_name="Jira")],
        actors=(("B7", "Jira"),)))
    assert contents(stream)[1].startswith(f"[{STAMP0} Jira (other_bot) id=B7 ")


# ------------------------------------------------------------------ body

def test_body_line_starting_with_a_bracket_is_escaped_once():
    stream = serialize_stream(pinned([msg(T0, text="[not a marker]\nplain\n[also]")]))
    body = contents(stream)[1].split("\n")[1:]
    assert body == ["\\[not a marker]", "plain", "\\[also]"]


def test_empty_body_renders_header_only():
    stream = serialize_stream(pinned([msg(T0, text="   ")]))
    assert contents(stream)[1] == f"[{STAMP0} alice (human) id=U1 ts={T0}]"


def test_mentions_render_from_the_frozen_actor_map():
    m = msg(T0, text="<@U2> and <@U3|label> and <@U9>", mentions=("U2", "U3", "U9"))
    stream = serialize_stream(pinned([m], actors=(("U1", "alice"), ("U2", "bob"),
                                                  ("U3", "carol"))))
    assert contents(stream)[1].endswith("@bob and @carol and @U9")


def test_no_trailing_newline_on_an_item():
    stream = serialize_stream(pinned([msg(T0, text="x", reactions=[ReactionRec("eyes", 1)])]))
    assert not contents(stream)[1].endswith("\n")


# ------------------------------------------------------------------ trailing markers

def test_files_marker_bytes_singular_and_plural():
    one = msg(T0, text="see this", files=[FileRef("F1", "a.png", "image/png", 9, None, "image")])
    assert contents(serialize_stream(pinned([one])))[1].split("\n")[-1] == (
        "[+1 file: a.png (image) id=F1]")
    two = msg(T0, text="see", files=[
        FileRef("F1", "a.png", "image/png", 9, None, "image"),
        FileRef(None, "b.csv", "text/csv", 9, None, "file")])
    assert contents(serialize_stream(pinned([two])))[1].split("\n")[-1] == (
        "[+2 files: a.png (image) id=F1, b.csv (file) id=unknown]")


def test_files_marker_reports_the_true_count_past_the_render_cap():
    """The count is the truth; the omission is declared. "+10 files" on a message carrying twelve
    would be a line the model cannot check and has no reason to disbelieve."""
    many = msg(T0, text="a pile", files=[
        FileRef(f"F{i}", f"f{i}.csv", "text/csv", 9, None, "file") for i in range(12)])
    marker = contents(serialize_stream(pinned([many])))[1].split("\n")[-1]
    assert marker.startswith("[+12 files: f0.csv (file) id=F0, ")
    assert marker.endswith(", +2 more not listed]")
    assert marker.count("id=") == 10


def test_files_marker_bytes_are_unchanged_at_the_cap():
    ten = msg(T0, text="ten", files=[
        FileRef(f"F{i}", f"f{i}.csv", "text/csv", 9, None, "file") for i in range(10)])
    marker = contents(serialize_stream(pinned([ten])))[1].split("\n")[-1]
    assert marker.startswith("[+10 files: ")
    assert "not listed" not in marker


def test_reactions_marker_takes_top_two_by_count_then_name():
    m = msg(T0, text="x", reactions=[ReactionRec("zzz", 3), ReactionRec("aaa", 3),
                                     ReactionRec("bbb", 9)])
    assert contents(serialize_stream(pinned([m])))[1].split("\n")[-1] == (
        "[reactions: 9× bbb, 3× aaa]")


def test_marker_order_is_files_then_sidecars_then_reactions():
    m = msg(T0, text="body", files=[FileRef("F1", "a.png", "image/png", 1, None, "image")],
            reactions=[ReactionRec("eyes", 1)])
    cards = sidecars(documents=[{"message_ts": T0, "filename": "spec.pdf",
                                "summary": "a real summary"}])
    lines = contents(serialize_stream(pinned([m], cards=cards)))[1].split("\n")
    assert lines[1] == "body"
    assert lines[2].startswith("[+1 file:")
    assert lines[3] == "[document (spec.pdf): summary available]"
    assert lines[4].startswith("[reactions:")


def test_marker_lines_are_exempt_from_the_body_bracket_escape():
    cards = sidecars(documents=[{"message_ts": T0, "filename": "x.pdf", "summary": "s"}])
    lines = contents(serialize_stream(pinned([msg(T0, text="hi")], cards=cards)))[1]
    assert "\\[document" not in lines
    assert "[document (x.pdf): summary available]" in lines


# ------------------------------------------------------------------ sidecar kinds

def test_document_marker_skips_the_unattended_placeholder_and_empty_summaries():
    cards = sidecars(documents=[
        {"message_ts": T0, "filename": "read.pdf", "summary": "real"},
        {"message_ts": T0, "filename": "unread.csv",
         "summary": "Shared in this conversation (unread.csv). Not yet read."},
        {"message_ts": T0, "filename": "blank.txt", "summary": "  "},
    ])
    item = contents(serialize_stream(pinned([msg(T0)], cards=cards)))[1]
    assert "[document (read.pdf): summary available]" in item
    assert "unread.csv" not in item
    assert "blank.txt" not in item


def test_image_marker_bytes_flatten_and_cap_the_gist():
    cards = sidecars(images=[{"message_ts": T0, "url": "https://x/files-pri/T-F42/a.png",
                              "analysis": "line one\nline two", "metadata": None}])
    item = contents(serialize_stream(pinned([msg(T0)], cards=cards)))[1]
    assert "[image analysis (F42): line one line two]" in item
    long_cards = sidecars(images=[{"message_ts": T0, "url": "u", "analysis": "z" * 400,
                                   "metadata": {"filename": "shot.png"}}])
    item = contents(serialize_stream(pinned([msg(T0)], cards=long_cards)))[1]
    assert item.endswith("[image analysis (shot.png): " + "z" * 200 + "]")


def test_image_marker_is_omitted_without_a_stored_analysis():
    cards = sidecars(images=[{"message_ts": T0, "url": "u", "analysis": "", "metadata": None}])
    assert "image analysis" not in contents(serialize_stream(pinned([msg(T0)], cards=cards)))[1]


def test_ambient_marker_requires_ready_and_excludes_unfurls():
    ready = {"source_ts": T0, "kind": "link", "ref": "https://a", "title": "Post",
             "summary": "what it says", "status": "ready", "derivation_source": "fetch"}
    unfurl = dict(ready, ref="https://b", derivation_source="unfurl")
    pending = dict(ready, ref="https://c", status="pending")
    item = contents(serialize_stream(pinned(
        [msg(T0)], cards=sidecars(ambient=[ready, unfurl, pending]))))[1]
    assert "[link content: Post — what it says]" in item
    assert item.count("[link") == 1


def test_tool_provenance_marker_renders_from_pinned_rows():
    cards = sidecars(tools=[(T0, ({"tool_name": "web_search", "gist": "q=weather"},))])
    item = contents(serialize_stream(pinned(
        [msg(T0, sender_type="self", sender="B0")],
        cards=SidecarPin(**{**cards.__dict__,
                            "receipts": (ReceiptRec(T0, "finalized", "t1", None),)}))))[1]
    assert "[used tools: web_search" in item


def test_sidecar_markers_are_deduped_on_channel_source_kind_and_ref():
    row = {"message_ts": T0, "filename": "dup.pdf", "summary": "one"}
    cards = sidecars(documents=[row, dict(row, summary="two")])
    item = contents(serialize_stream(pinned([msg(T0)], cards=cards)))[1]
    assert item.count("[document (dup.pdf)") == 1


def test_sidecar_markers_sort_by_kind_then_ref():
    cards = sidecars(
        documents=[{"message_ts": T0, "filename": "zz.pdf", "summary": "s"},
                   {"message_ts": T0, "filename": "aa.pdf", "summary": "s"}],
        ambient=[{"source_ts": T0, "kind": "link", "ref": "https://a", "title": "T",
                  "summary": "s", "status": "ready", "derivation_source": "fetch"}],
        images=[{"message_ts": T0, "url": "u", "analysis": "a", "metadata": None}])
    lines = contents(serialize_stream(pinned([msg(T0, text="b")], cards=cards)))[1].split("\n")
    assert lines[2].startswith("[link content:")          # ambient
    assert lines[3] == "[document (aa.pdf): summary available]"
    assert lines[4] == "[document (zz.pdf): summary available]"
    assert lines[5].startswith("[image analysis")


# ------------------------------------------------------------------ roles + receipts

def test_other_senders_are_always_role_user():
    stream = serialize_stream(pinned([msg(T0), msg(T1, sender_type="other_bot", sender="B2")]))
    assert [i.role for i in stream.message_items] == ["user", "user"]
    assert stream.receipts_included == ()
    assert stream.receipts_excluded == ()


def test_finalized_receipt_makes_a_self_message_the_assistant():
    cards = sidecars(receipts=[ReceiptRec(T0, "finalized", "turn-1", None)])
    stream = serialize_stream(pinned([msg(T0, sender="B0", sender_type="self", text="my reply")],
                                     cards=cards))
    assert stream.message_items[0].role == "assistant"
    assert stream.receipts_included == (T0,)
    assert stream.receipts_excluded == ()


@pytest.mark.parametrize("state", ["in_flight", "chrome"])
def test_in_flight_and_chrome_are_excluded_from_the_stream(state):
    cards = sidecars(receipts=[ReceiptRec(T0, state, "turn-1", None)])
    stream = serialize_stream(pinned([msg(T0, sender="B0", sender_type="self", text="half")],
                                     cards=cards))
    assert stream.message_items == ()
    assert stream.receipts_excluded == (T0,)
    assert stream.receipts_included == ()


def test_a_post_epoch_self_message_with_no_receipt_is_excluded(caplog):
    """The receipt invariant, fail-closed. A post-epoch message with no row is evidence that a
    registration was LOST, not evidence of a reply — and the shape heuristic cannot tell an answer
    from a status card well enough to attribute words to ourselves on that basis."""
    cards = sidecars(epoch=T0)
    real = msg(T1, sender="B0", sender_type="self", text="a genuine-looking answer")
    with caplog.at_level("WARNING"):
        stream = serialize_stream(pinned([real], cards=cards))
    assert stream.message_items == ()
    assert stream.receipts_excluded == (T1,)
    assert stream.receipts_included == ()
    assert "no receipt row" in caplog.text          # the tripwire stays loud


@pytest.mark.parametrize("epoch", [None, "", "not-a-ts"])
def test_a_missing_or_malformed_epoch_cannot_enable_the_heuristic(epoch, caplog):
    """Without a parseable epoch there is no proof the message predates receipts, so it is
    treated as post-epoch and excluded. The direction that fails closed is the one that omits a
    message we posted, not the one that replays chrome as our own past words."""
    cards = sidecars(epoch=epoch)
    with caplog.at_level("WARNING"):
        stream = serialize_stream(pinned(
            [msg(T1, sender="B0", sender_type="self", text="looks like an answer")],
            cards=cards))
    assert stream.message_items == ()
    assert stream.receipts_excluded == (T1,)


def test_a_finalized_receipt_still_admits_a_post_epoch_message():
    cards = sidecars(epoch=T0, receipts=[ReceiptRec(T1, "finalized", "turn-1", None)])
    stream = serialize_stream(pinned(
        [msg(T1, sender="B0", sender_type="self", text="my reply")], cards=cards))
    assert [i.role for i in stream.message_items] == ["assistant"]
    assert stream.receipts_included == (T1,)


def test_pre_epoch_self_message_is_grandfathered_by_shape():
    cards = sidecars(epoch=T2)
    real = msg(T0, sender="B0", sender_type="self", text="a genuine answer")
    chrome = msg(T1, sender="B0", sender_type="self", text=":hourglass: Thinking...")
    stream = serialize_stream(pinned([real, chrome], cards=cards))
    assert [i.metadata["ts"] for i in stream.message_items] == [T0]
    assert stream.receipts_included == (T0,)
    assert stream.receipts_excluded == (T1,)


def test_self_message_with_no_text_and_no_files_is_never_replayed():
    stream = serialize_stream(pinned([msg(T0, sender="B0", sender_type="self", text="")]))
    assert stream.message_items == ()
    assert stream.receipts_excluded == (T0,)


def test_membership_hash_is_the_published_canonicalization():
    cards = sidecars(receipts=[ReceiptRec(T0, "finalized", "t", None),
                               ReceiptRec(T1, "in_flight", "t", None)])
    stream = serialize_stream(pinned([msg(T0, sender="B0", sender_type="self", text="a"),
                                      msg(T1, sender="B0", sender_type="self", text="b")],
                                     cards=cards))
    expected = hashlib.sha256(
        (f"included:{T0};excluded:{T1}").encode("utf-8")).hexdigest()
    assert stream.receipts_membership_hash == expected


def test_membership_hash_sorts_numerically_not_lexicographically():
    wide, narrow = "1700000000.000010", "1700000000.0002"
    cards = sidecars(receipts=[ReceiptRec(wide, "finalized", "t", None),
                               ReceiptRec(narrow, "finalized", "t", None)])
    stream = serialize_stream(pinned([msg(narrow, sender="B0", sender_type="self", text="b"),
                                      msg(wide, sender="B0", sender_type="self", text="a")],
                                     cards=cards))
    # v3 renders the periphery in TS ORDER, so membership follows the rendered order.
    assert stream.receipts_included == (wide, narrow)
    expected = hashlib.sha256(
        f"included:{wide},{narrow};excluded:".encode("utf-8")).hexdigest()
    assert stream.receipts_membership_hash == expected


# ------------------------------------------------------------------ metadata + hash

def test_item_metadata_keys_are_exactly_the_pinned_set():
    stream = serialize_stream(pinned([msg(T1, root=T0)]))
    assert set(stream.message_items[0].metadata) == {
        "channel_id", "sender_id", "ts", "thread_root_ts", "sender_type"}
    assert stream.message_items[0].metadata["ts"] == T1
    assert stream.message_items[0].metadata["thread_root_ts"] == T0


def test_hash_is_role_newline_content_nul_per_item_in_order():
    stream = serialize_stream(pinned([msg(T0, text="one"), msg(T1, text="two")]))
    digest = hashlib.sha256()
    for item in stream.items:
        digest.update(f"{item.role}\n{item.content}\x00".encode("utf-8"))
    assert stream.stream_sha256 == digest.hexdigest()


def test_hash_changes_when_any_rendered_byte_changes():
    a = serialize_stream(pinned([msg(T0, text="one")])).stream_sha256
    b = serialize_stream(pinned([msg(T0, text="onE")])).stream_sha256
    assert a != b


def test_byte_count_counts_utf8_content_bytes():
    stream = serialize_stream(pinned([msg(T0, text="é")]))
    assert stream.byte_count == sum(len(i.content.encode("utf-8")) for i in stream.items)


# ------------------------------------------------------------------ selectors

def test_origin_slice_and_trigger_view_select_after_serialization():
    """`origin_slice` KEEPS ITS NAME AND SIGNATURE but its meaning changed: the origin is now
    FETCHED into its own post-breakpoint block rather than selected out of the window, so it
    returns that block for the origin this build was made for and `()` for anything else."""
    root = msg(T0, text="root")
    reply = msg(T1, text="reply", root=T0, origin=ORIGIN_REPLIES)
    other = msg(T2, text="elsewhere")
    pin = pinned([root, reply, other])
    object.__setattr__(pin, "origin_root_ts", T0)
    object.__setattr__(pin, "origin_snapshot", (root, reply))
    stream = serialize_stream(pin)

    assert [i.metadata["ts"] for i in stream.origin_slice(T0)] == [T0, T1]
    assert stream.origin_slice(T2) == (), "a foreign root is not this build's origin"
    assert stream.origin_slice(None) == ()
    assert stream.trigger_view(T2).metadata["ts"] == T2
    assert stream.trigger_view("1600000000.000000") is None


def test_normalized_for_returns_the_pinned_record():
    stream = serialize_stream(pinned([msg(T0, text="x")]))
    assert stream.normalized_for(T0).text == "x"
    assert stream.normalized_for("1.0") is None


# ------------------------------------------------------------------ end marker + assembly

def test_end_marker_breakpoint_is_attached_after_api_part():
    stream = serialize_stream(pinned([msg(T0)]))
    part = stream.end_marker_content("gpt-5.6-sol")[0]
    assert part["type"] == "input_text"
    assert part["text"] == END_MARKER_TEXT
    assert part["prompt_cache_breakpoint"] == {"mode": "explicit"}


def test_end_marker_breakpoint_is_dropped_on_a_model_without_explicit_breakpoints():
    stream = serialize_stream(pinned([msg(T0)]))
    assert "prompt_cache_breakpoint" not in stream.end_marker_content("gpt-5.5")[0]


def test_input_items_strip_metadata_before_send():
    stream = serialize_stream(pinned([msg(T0)]))
    assert all(set(item) == {"role", "content"} for item in stream.input_items())


def test_channel_layout_builder_strips_metadata(monkeypatch):
    from openai_client.base import _build_request_params
    stream = serialize_stream(pinned([msg(T0)]))
    items = [{**i, "metadata": {"ts": T0}} for i in stream.input_items()]
    params = _build_request_params(model="gpt-5.6-sol", input_items=items, layout="channel")
    assert all("metadata" not in item for item in params["input"])


# ------------------------------------------------------------------ dedup

def test_dedup_prefers_the_replies_record_whole():
    history_copy = msg(T1, text="broadcast", root=T0, broadcast=True, origin=ORIGIN_HISTORY)
    replies_copy = msg(T1, text="broadcast", root=T0, broadcast=True, origin=ORIGIN_REPLIES)
    for order in ([history_copy, replies_copy], [replies_copy, history_copy]):
        kept = _dedup(order)
        assert len(kept) == 1
        assert kept[0].origin == ORIGIN_REPLIES


def test_dedup_sorts_numerically():
    wide, narrow = "1700000000.000010", "1700000000.0002"
    kept = _dedup([msg(narrow), msg(wide)])
    assert [m.ts for m in kept] == [wide, narrow]


def test_broadcast_renders_once_with_its_flag():
    root = msg(T0, text="root")
    stream = serialize_stream(pinned(_dedup([
        root,
        msg(T1, text="cast", root=T0, broadcast=True, origin=ORIGIN_HISTORY),
        msg(T1, text="cast", root=T0, broadcast=True, origin=ORIGIN_REPLIES)])))
    assert stream.message_count == 2
    assert stream.message_items[1].content.split("\n")[0].endswith(" (broadcast)]")


# ------------------------------------------------------------------ determinism

def test_two_independent_builds_of_the_same_pins_are_byte_identical():
    messages = [msg(T0, text="root"), msg(T1, text="reply", root=T0, origin=ORIGIN_REPLIES),
                msg(T2, text="later", sender="U2")]
    cards = sidecars(documents=[{"message_ts": T0, "filename": "a.pdf", "summary": "s"}])
    first = serialize_stream(pinned(list(messages), cards=cards))
    second = serialize_stream(pinned(list(messages), cards=cards))
    assert contents(first) == contents(second)
    assert first.stream_sha256 == second.stream_sha256
    assert first.receipts_membership_hash == second.receipts_membership_hash


def test_serializer_config_snapshot_is_hashed_material_not_a_constant(monkeypatch):
    from config import config
    before = serializer_config_snapshot()
    monkeypatch.setattr(config, "tool_provenance_line_budget", 999, raising=False)
    assert serializer_config_snapshot() != before


# ------------------------------------------------------------------ pin purity + immutability

def test_the_pin_carries_the_frozen_config_values_not_just_a_hash():
    pin = pinned([msg(T0)])
    assert pin.serializer_config["image_gist_chars"] == 200
    assert isinstance(pin.serializer_config.get("chrome_markers"), tuple)


def test_serialization_reads_no_live_config(monkeypatch):
    """Serializing the same pin twice with the world changed in between must produce the same
    bytes: everything dynamic was resolved when the tuple was pinned, and a config change is a
    cache MISS through serializer_config_hash rather than a different stream under one hash."""
    from config import config
    cards = sidecars(
        images=[{"message_ts": T0, "url": "u", "analysis": "z" * 400, "metadata": None}],
        ambient=[{"source_ts": T0, "kind": "link", "ref": "https://a", "title": "T",
                  "summary": "s" * 500, "status": "ready", "derivation_source": "fetch"}],
        tools=[(T0, ({"tool_name": "web_search", "gist": "q=weather"},))])
    pin = pinned([msg(T0, sender_type="self", sender="B0"),
                  msg(T1, text="x", reactions=[ReactionRec("a", 3), ReactionRec("b", 2),
                                               ReactionRec("c", 1)])],
                 cards=SidecarPin(**{**cards.__dict__,
                                     "receipts": (ReceiptRec(T0, "finalized", "t1", None),)}))
    before = serialize_stream(pin)

    monkeypatch.setattr(config, "tool_provenance_line_budget", 5, raising=False)
    monkeypatch.setattr(config, "tool_provenance_gist_chars", 1, raising=False)
    after = serialize_stream(pin)

    assert contents(after) == contents(before)
    assert after.stream_sha256 == before.stream_sha256


def test_the_pinned_caps_are_the_ones_the_serializer_applies():
    cfg = dict(serializer_config_snapshot(), files_marker_limit=1, reactions_rendered=1,
               image_gist_chars=5)
    cards = sidecars(images=[{"message_ts": T0, "url": "u", "analysis": "abcdefghij",
                              "metadata": {"filename": "shot.png"}}])
    item = contents(serialize_stream(pinned(
        [msg(T0, text="x", files=[FileRef("F1", "a.csv", "text/csv", 1, None, "file"),
                                  FileRef("F2", "b.csv", "text/csv", 1, None, "file")],
             reactions=[ReactionRec("aa", 2), ReactionRec("bb", 1)])],
        cards=cards, serializer_config=cfg)))[1]
    assert "[+2 files: a.csv (file) id=F1, +1 more not listed]" in item
    assert "[image analysis (shot.png): abcde]" in item
    assert "[reactions: 2× aa]" in item


def test_pinned_sidecar_rows_and_item_metadata_are_read_only():
    cards = sidecars(documents=[{"message_ts": T0, "filename": "a.pdf", "summary": "s"}])
    stream = serialize_stream(pinned([msg(T0)], cards=cards))
    with pytest.raises(TypeError):
        stream.pinned.sidecars.document_extractions[0]["summary"] = "rewritten"
    with pytest.raises(TypeError):
        stream.message_items[0].metadata["ts"] = "1.0"
    with pytest.raises(TypeError):
        stream.pinned.serializer_config["image_gist_chars"] = 1


def test_a_mutated_source_row_cannot_change_the_pinned_bytes():
    row = {"message_ts": T0, "filename": "a.pdf", "summary": "s"}
    pin = pinned([msg(T0)], cards=sidecars(documents=[row]))
    before = serialize_stream(pin).stream_sha256
    row["summary"] = ""                     # the caller's dict, not the pin's copy
    row["filename"] = "renamed.pdf"
    assert serialize_stream(pin).stream_sha256 == before


# ------------------------------------------------------------------ malformed timestamps

@pytest.mark.parametrize("rows", [
    {"documents": [{"message_ts": "later", "filename": "a.pdf", "summary": "s"}]},
    {"images": [{"message_ts": None, "url": "u", "analysis": "a", "metadata": None}]},
    {"ambient": [{"source_ts": "", "kind": "link", "ref": "r", "title": "T", "summary": "s",
                  "status": "ready", "derivation_source": "fetch"}]},
    {"tools": [("nonsense", ({"tool_name": "web_search"},))]},
])
def test_a_sidecar_row_with_an_unusable_ts_fails_the_turn_at_freeze_time(rows):
    """Validated when it is frozen, not when a renderer happens to reach it: a row we cannot
    place in time would otherwise be dropped by whichever consumer looked first."""
    with pytest.raises(StreamTimestampError):
        sidecars(**rows)


def test_a_receipt_with_an_unusable_ts_fails_the_turn():
    with pytest.raises(StreamTimestampError):
        sidecars(receipts=[ReceiptRec("soon", "finalized", "t", None)])


def test_a_malformed_h_or_floor_fails_the_pin():
    with pytest.raises(StreamTimestampError):
        pinned([msg(T0)], h="tomorrow")


# ------------------------------------------------------------------ T6: the v3 sequence

def test_the_canonical_sequence_is_horizon_messages_marker():
    """T6. v3's whole item vocabulary: the horizon, the messages, the end marker, and nothing
    else. No summary item is PRODUCIBLE — the pin has no field that could carry one — so a
    channel cannot render compacted history it no longer has."""
    cards = sidecars(receipts=[ReceiptRec(ts=T1, state="finalized", turn_id="t1",
                                          thread_root_ts=None)])
    stream = serialize_stream(pinned(
        [msg(T0, text="a human said this"),
         msg(T1, text="and we answered", sender="B0", sender_type="self")], cards=cards))
    assert SERIALIZER_VERSION == 3
    assert stream.items[0] is stream.horizon_item
    assert stream.items[-1] is stream.end_marker_item
    assert stream.items[1:-1] == stream.message_items
    assert stream.message_count == 2
    # The v1 role rule, unchanged: framing is always `user`, and one of OUR OWN messages with a
    # FINALIZED receipt is `assistant`. Making every item `user` would erase the model's own
    # past turns from its own history.
    assert [item.role for item in stream.items] == [
        "user",        # horizon
        "user",        # the human
        "assistant",   # our own finalized reply
        "user",        # end marker
    ]
    assert not hasattr(stream, "summary_item")
    assert not any("CHANNEL SUMMARY" in item.content for item in stream.items)


def test_the_hash_keeps_the_v1_framing_over_the_v3_sequence():
    stream = serialize_stream(pinned([msg(T0)]))
    digest = hashlib.sha256()
    for item in stream.items:
        digest.update(f"{item.role}\n{item.content}\x00".encode("utf-8"))
    assert stream.stream_sha256 == digest.hexdigest()
    # The end marker participates; its model-specific breakpoint decoration does not.
    assert stream.items[-1].content == END_MARKER_TEXT
    assert "prompt_cache_breakpoint" in stream.end_marker_content("gpt-5.6-sol")[0]


def test_the_warm_horizon_is_pinned_LITERALLY_end_to_end():
    """A1's whole item, spelled out — NOT built from `HORIZON_TEMPLATE`.

    Every other horizon assertion here interpolates the same constant the renderer uses, so a
    third line (a reach clause, a search hint, a count) could be added and every one of them
    would still pass. This is the only test that would fail, and it is the one that pins A1's
    empty-`reach_tools` rule: with no reach tools the segment "; older history exists and is
    reachable with …" is ABSENT, not empty — the model is never told history is reachable by a
    tool it does not have.
    """
    stream = serialize_stream(pinned([msg(T0)]))
    assert stream.horizon_item.content == (
        f"[STREAM HORIZON: the recent activity in this channel, from {COVERAGE.start_ts}]\n"
        "Images and code-execution results in this stream are awareness-only outside the "
        "current thread; current-thread file, image and container details follow after the "
        "stream."
    )
    # Stated as absences too, so the failure message names WHICH half regressed.
    assert "older history exists" not in stream.horizon_item.content
    assert "search_slack" not in stream.horizon_item.content
    assert len(stream.horizon_item.content.split("\n")) == 2


# ------------------------------------------------------------------ §2f: the six index states

@pytest.mark.parametrize("status,reason,state", [
    ("pending", None, "cold"),
    ("running", "genesis", "cold"),
    ("complete", "genesis", "warm"),
    ("limited", "retention", "limited_retention"),
    ("limited", "depth_config", "limited_depth"),
    ("limited", "unavailable", "unavailable"),
    ("limited", None, "limited_retention"),          # the fail-safe
    ("limited", "something new", "limited_retention"),
])
def test_the_inventory_state_table_is_complete(status, reason, state):
    assert InventoryPin(start_ts=COVERAGE.start_ts, status=status, reason=reason).state == state
    assert state in INVENTORY_STATES


@pytest.mark.parametrize("state,fragment", [
    ("absent", "have not indexed this channel's older threads yet"),
    ("cold", "have not indexed this channel's older threads yet"),
    ("warm", ""),
    ("limited_retention", "Slack's retention limits how far back my thread index reaches"),
    ("limited_depth", "my thread index reaches back 90 days"),
    ("unavailable", "could not index this channel's older threads"),
])
def test_every_state_renders_exactly_one_index_clause(state, fragment):
    clause = index_clause(state, depth_days=90)
    assert clause == "" if not fragment else fragment in clause


def test_an_unavailable_index_renders_a_clause_instead_of_failing_the_turn():
    """§2f correction 1. Under OWNER-7 the sweep builds a thread INDEX, so failing to build it
    degrades discovery and says so — it no longer defines the stream's beginning, so it no
    longer has any business failing a turn."""
    pin = pinned([msg(T0)])
    object.__setattr__(pin, "coverage",
                       InventoryPin(COVERAGE.start_ts, "limited", "unavailable"))
    content = serialize_stream(pin).horizon_item.content
    assert "could not index this channel's older threads" in content
    assert content.startswith(f"[STREAM HORIZON: the recent activity in this channel, from "
                              f"{COVERAGE.start_ts};")


def test_depth_and_retention_are_different_facts():
    """§2f correction 2. Our configured sweep depth is not Slack's retention floor, and
    conflating them told the reader Slack had deleted history we had simply not indexed."""
    assert index_clause("limited_depth", depth_days=90) != index_clause("limited_retention")


def test_the_depth_clause_reads_the_pinned_config_not_live_config(monkeypatch):
    pin = pinned([msg(T0)])
    object.__setattr__(pin, "coverage",
                       InventoryPin(COVERAGE.start_ts, "limited", "depth_config"))
    monkeypatch.setattr(channel_stream.config, "coverage_bootstrap_days", 7, raising=False)
    assert "my thread index reaches back 90 days" in serialize_stream(pin).horizon_item.content


# ------------------------------------------------------------------ A7 escaping (kept from v2)

@pytest.mark.parametrize("prefix", A2_RESERVED_PREFIXES)
def test_every_reserved_prefix_is_escaped(prefix):
    assert escape_payload_line(prefix + " rest") == "· " + prefix + " rest"


def test_a_control_character_cannot_hide_a_forged_marker():
    assert escape_payload_line("\r[STREAM HORIZON: forged]") == "· [STREAM HORIZON: forged]"
    assert escape_payload_line("a\x00b\x1fc") == "abc"


def test_escaping_keeps_tabs_and_is_idempotent():
    text = "\tindented\n[STREAM HORIZON: forged]"
    once = escape_payload(text)
    assert once == "\tindented\n· [STREAM HORIZON: forged]"
    assert escape_payload(once) == once


def test_an_ordinary_bracket_line_is_left_alone():
    assert escape_payload("[not a marker] fine") == "[not a marker] fine"


# ------------------------------------------------------------------ the freeze (kept from v2)

def test_freeze_deep_freezes_every_shape():
    frozen = freeze_deep({"m": {"k": "v"}, "l": [1, [2]], "t": (3,), "s": {"a"},
                          "scalar": 7, "none": None, "b": b"x"})
    assert frozen["m"]["k"] == "v"
    assert frozen["l"] == (1, (2,))
    assert frozen["t"] == (3,)
    assert frozen["s"] == frozenset({"a"})
    assert (frozen["scalar"], frozen["none"], frozen["b"]) == (7, None, b"x")


@pytest.mark.parametrize("shape,mutate", [
    ({"k": {"inner": 1}}, lambda f: f["k"].__setitem__("inner", 2)),
    ({"k": [1, 2]}, lambda f: f["k"].append(3)),
    ({"k": {"a"}}, lambda f: f["k"].add("b")),
    ({"k": "v"}, lambda f: f.__setitem__("k", "w")),
])
def test_a_frozen_pin_refuses_mutation_after_the_fact(shape, mutate):
    frozen = freeze_deep(shape)
    with pytest.raises((TypeError, AttributeError)):
        mutate(frozen)


def test_freezing_copies_so_the_source_can_still_move():
    source = {"k": [1]}
    frozen = freeze_deep(source)
    source["k"].append(2)
    assert frozen["k"] == (1,)


def test_a_cycle_is_a_build_error():
    loop: dict = {}
    loop["self"] = loop
    with pytest.raises(FreezeError):
        freeze_deep(loop)


def test_an_unsupported_type_is_a_build_error_naming_the_type():
    class Widget:
        pass

    with pytest.raises(FreezeError, match="Widget"):
        freeze_deep({"k": Widget()})


def test_item_metadata_is_frozen():
    stream = serialize_stream(pinned([msg(T0)]))
    with pytest.raises(TypeError):
        stream.message_items[0].metadata["ts"] = "9.0"


def test_root_snippet_semantics_are_unchanged():
    assert _root_snippet(msg(T0, tombstone=True)) == '~"[deleted]"'
    assert _root_snippet(msg(T0, text="one two three four five six seven")) == (
        '~"one two three four five six"')
    assert _root_snippet(None) == ""


# ------------------------------------------------- the channel with no inventory row at all

def test_an_absent_inventory_row_renders_rather_than_refusing():
    """§2f, as ruled for W1: nothing about the inventory fails a turn. With no row the fetch runs
    unfloored, so the horizon names the oldest message this build ACTUALLY holds — never the
    SELECTED floor, and the inventory state rides beside it as its own clause."""
    pin = pinned([msg(T0), msg(T1)])
    object.__setattr__(pin, "coverage", None)
    object.__setattr__(pin, "periphery_floor_ts", T0)
    object.__setattr__(pin, "window", (T0, True))
    stream = serialize_stream(pin)

    assert pin.inventory_state == "absent"
    assert pin.horizon_floor_ts == T0
    assert stream.horizon_item.content.startswith(
        f"[STREAM HORIZON: the recent activity in this channel, from {T0};")
    assert "have not indexed this channel's older threads yet" in stream.horizon_item.content
    assert '"0"' not in stream.horizon_item.content
    assert stream.message_count == 2


def test_an_empty_build_with_no_row_omits_the_floor_clause_entirely():
    """A1's no-floor variant. No row and nothing fetched means there is no message whose ts the
    window could honestly claim to begin at, so the clause is dropped rather than filled with a
    sentinel the model would read as a timestamp."""
    pin = pinned([])
    object.__setattr__(pin, "coverage", None)
    object.__setattr__(pin, "periphery_floor_ts", "")
    object.__setattr__(pin, "window", ("", True))
    stream = serialize_stream(pin)

    assert pin.horizon_floor_ts == ""
    assert stream.horizon_item.content.startswith(
        "[STREAM HORIZON: no recent messages in this channel;")
    assert stream.items == (stream.horizon_item, stream.end_marker_item)
    assert stream.stream_render_fields()["inventory_start_ts"] == ""
    assert stream.stream_render_fields()["inventory_state"] == "absent"


def test_an_empty_window_still_renders_its_framing():
    """A floor above H is an empty window, not a failure — the horizon and the end marker are
    always present, which is what makes 'the stream said nothing' distinguishable from 'the
    stream was never built'."""
    stream = serialize_stream(pinned([]))
    assert stream.items == (stream.horizon_item, stream.end_marker_item)
    assert stream.message_count == 0
    assert stream.horizon_item.content == HORIZON


# ================================================== W2: the shallow window's serializer

def origin_pinned(periphery, origin_messages, *, origin_root, cards=None, floor=None,
                  reach=("search_slack", "fetch_channel_history", "fetch_thread_messages"),
                  actors=(("U1", "alice"),), h=T2):
    """A pin carrying BOTH blocks. The origin is fetched, not selected out of the window."""
    cards = cards if cards is not None else sidecars()
    resolved_floor = cards.window[0] if floor is None else floor
    config = serializer_config_snapshot()
    return PinnedTuple(
        team_id=TEAM, channel_id=CH, window=(resolved_floor, True), H=h,
        periphery_floor_ts=resolved_floor, selection_version=1, reach_tools=tuple(reach),
        chrome_ts=classify_chrome(tuple(periphery) + tuple(origin_messages),
                                  chrome_markers=config["chrome_markers"]),
        fetch_snapshot=tuple(periphery), origin_root_ts=origin_root,
        origin_snapshot=tuple(origin_messages),
        sidecar_versions_hash=cards.versions_hash,
        actor_map=tuple(actors), actor_map_hash="actorhash",
        serializer_version=SERIALIZER_VERSION, serializer_config_hash="cfghash",
        capability_profile_hash="caphash", tool_schema_version="tools-1",
        coverage=COVERAGE, receipt_feature_epoch_ts=cards.receipt_feature_epoch_ts,
        receipt_map=tuple((r.ts, r.state, r.turn_id, r.thread_root_ts) for r in cards.receipts),
        sidecars=cards, serializer_config=config)


def test_an_origin_message_inside_the_periphery_appears_twice():
    """T33. RULING-1's duplication, pinned as the CONTRACT rather than tolerated.

    An origin message inside the window renders once among its ts-neighbours in the shared
    prefix and once inside the complete origin block. De-duplicating it would either fork the
    prefix per origin or leave a hole in the thread — the header says so in words instead."""
    root = msg(T0, text="the origin root")
    stream = serialize_stream(origin_pinned([root], [root], origin_root=T0))

    assert [i.metadata["ts"] for i in stream.message_items] == [T0]
    assert [i.metadata["ts"] for i in stream.origin_items] == [T0]
    assert "that is the same thread seen from the room" in stream.origin_header_item.content


def test_a_broadcast_yields_one_periphery_event():
    """T34. Broadcast dedup unchanged, the replies copy winning."""
    history_copy = msg(T1, text="broadcast", root=T0, broadcast=True, origin=ORIGIN_HISTORY)
    replies_copy = msg(T1, text="broadcast", root=T0, broadcast=True, origin=ORIGIN_REPLIES)
    stream = serialize_stream(pinned(_dedup([history_copy, replies_copy])))

    assert stream.message_count == 1
    # THE REPLIES COPY WON — which is the half that matters: the history copy of a broadcast
    # carries the root's shape, so keeping it would turn a reply into a root.
    assert stream.normalized_for(T1).origin == ORIGIN_REPLIES
    # AFTER dedup a broadcast is the REPLY it is, never a root.
    assert stream.root_count == 0


def test_an_in_flight_own_message_is_excluded_from_both_blocks():
    """T36. Receipt-state exclusion applies to the ORIGIN too, and `receipts_excluded` names it.

    Showing the model its own half-finished sentence is how a stream turns into a hall of
    mirrors; the origin block is no more exempt from that than the periphery."""
    ours = msg(T1, text="half a reply", sender="B0", sender_type="self", root=T0)
    cards = sidecars(receipts=[ReceiptRec(ts=T1, state="in_flight", turn_id="t",
                                          thread_root_ts=T0)])
    stream = serialize_stream(origin_pinned([ours], [ours], origin_root=T0, cards=cards))

    assert stream.message_items == () and stream.origin_items == ()
    assert T1 in stream.receipts_excluded


def test_a_reply_whose_root_is_below_the_floor_gets_one_marker_item():
    """T42. The marker is a SEPARATE StreamItem, `role="user"`, `metadata={}`, positioned
    immediately before that root's FIRST reply. Two replies under one absent root ⇒ ONE marker."""
    absent = "1699000000.000000"
    first = msg(T1, text="first reply", root=absent)
    second = msg(T2, text="second reply", root=absent)
    stream = serialize_stream(pinned([first, second]))

    markers = [i for i in stream.items if i.content.startswith("[thread=")]
    assert len(markers) == 1, "a second reply under the same absent root adds nothing"
    assert markers[0].role == "user" and dict(markers[0].metadata) == {}
    assert stream.items.index(markers[0]) == stream.items.index(stream.message_items[0]) - 1
    assert stream.orphan_root_count == 1
    # The root is NOT pulled into the window, and it IS trusted — the label was rendered.
    assert absent not in [i.metadata["ts"] for i in stream.message_items]
    assert absent in stream.trusted_thread_roots


async def test_markers_are_origin_independent_and_counted_correctly():
    """T43. Three clauses: a present root gets NO marker; two origins built from ONE shared pin
    still render identical pre-breakpoint bytes WITH markers present; and marker items are
    content but not messages.

    The middle clause is the load-bearing one — markers are a function of the periphery
    selection, which is origin-independent by construction, so they live safely in the cached
    bytes. A marker that varied with the origin would fork the prefix on every thread.
    """
    root = msg(T0, text="present root")
    reply = msg(T1, text="reply", root=T0)
    present = serialize_stream(pinned([root, reply]))
    assert [i for i in present.items if i.content.startswith("[thread=")] == []
    assert present.orphan_root_count == 0

    absent = "1699000000.000000"
    shared = _shared_pin([msg(T1, text="reply", root=absent)])
    fetch_a = OriginFetch(origin_root_ts="1699500000.000000",
                          messages=(msg("1699500000.000000", text="thread A"),), pages=1,
                          empty_fallback=False, deadline_at=1000.0)
    fetch_b = OriginFetch(origin_root_ts="1699600000.000000",
                          messages=(msg("1699600000.000000", text="thread B"),), pages=1,
                          empty_fallback=False, deadline_at=1000.0)
    pin_a, _ = await build_origin_pin(shared, fetch_a, db=_PinDB())
    pin_b, _ = await build_origin_pin(shared, fetch_b, db=_PinDB())
    stream_a, stream_b = serialize_stream(pin_a), serialize_stream(pin_b)

    marker_a = [i for i in stream_a.items if i.content.startswith("[thread=")]
    assert len(marker_a) == 1, "the marker must be present for this clause to mean anything"
    assert [i.content for i in stream_a.items] == [i.content for i in stream_b.items]
    assert stream_a.stream_sha256 == stream_b.stream_sha256

    # Marker items count their BYTES but are not MESSAGES.
    orphan = serialize_stream(pinned([msg(T1, text="reply", root=absent)]))
    marker = [i for i in orphan.items if i.content.startswith("[thread=")][0]
    assert orphan.message_count == 1
    assert len(orphan.items) == 4            # horizon, marker, message, end marker
    assert orphan.byte_count >= len(marker.content.encode("utf-8"))


AWARENESS = (
    "Images and code-execution results in this stream are awareness-only outside the current "
    "thread; current-thread file, image and container details follow after the stream."
)


@pytest.mark.parametrize("status,reason,clause", [
    ("pending", None,
     "; I have not indexed this channel's older threads yet, so a recent reply under an older "
     "thread may be missing"),
    ("running", "genesis",
     "; I have not indexed this channel's older threads yet, so a recent reply under an older "
     "thread may be missing"),
    ("complete", "genesis", ""),
    ("limited", "retention",
     "; Slack's retention limits how far back my thread index reaches"),
    ("limited", "depth_config", "; my thread index reaches back 90 days"),
    ("limited", "unavailable", "; I could not index this channel's older threads"),
    ("limited", None, "; Slack's retention limits how far back my thread index reaches"),
])
def test_the_horizon_grammar_is_complete(status, reason, clause):
    """T52. LITERAL BYTES for every inventory state — not fragments. The row demands the exact
    item, because a fragment assertion passes against a horizon that grew an extra clause."""
    pin = pinned([msg(T0)])
    object.__setattr__(pin, "coverage", InventoryPin(COVERAGE.start_ts, status, reason))
    assert serialize_stream(pin).horizon_item.content == (
        f"[STREAM HORIZON: the recent activity in this channel, from {COVERAGE.start_ts}"
        f"{clause}]\n{AWARENESS}")


def test_an_absent_inventory_row_renders_the_unindexed_clause():
    """T52's SIXTH state — `absent`, which no `bootstrap_status` can produce because there is no
    row at all. It shares the `cold` wording deliberately, and it is reached by `coverage=None`
    rather than by any status value."""
    pin = pinned([msg(T0)])
    object.__setattr__(pin, "coverage", None)
    assert pin.inventory_state == "absent"
    assert serialize_stream(pin).horizon_item.content == (
        f"[STREAM HORIZON: the recent activity in this channel, from {COVERAGE.start_ts}"
        "; I have not indexed this channel's older threads yet, so a recent reply under an "
        f"older thread may be missing]\n{AWARENESS}")


def test_the_zero_roots_variant_omits_the_floor_clause_entirely():
    """T52's zero-roots case. With the empty-floor sentinel the floor clause is dropped rather
    than filled with a value the model would read as a timestamp."""
    pin = pinned([])
    object.__setattr__(pin, "periphery_floor_ts", "")
    object.__setattr__(pin, "window", ("", True))
    assert serialize_stream(pin).horizon_item.content == (
        f"[STREAM HORIZON: no recent messages in this channel]\n{AWARENESS}")


@pytest.mark.parametrize("reach,expected", [
    ((), ""),
    (("search_slack",), "; older history exists and is reachable with search_slack"),
    (("search_slack", "fetch_thread_messages"),
     "; older history exists and is reachable with search_slack and fetch_thread_messages"),
    (("search_slack", "fetch_channel_history", "fetch_thread_messages"),
     "; older history exists and is reachable with search_slack, fetch_channel_history and "
     "fetch_thread_messages"),
])
def test_the_horizon_reach_segment_names_exactly_the_exposed_tools(reach, expected):
    """T52's reach half. With NO reach tool the whole segment is ABSENT — the model is never
    told history is reachable by a tool it does not have."""
    pin = origin_pinned([msg(T0)], [], origin_root=None, reach=reach)
    line = serialize_stream(pin).horizon_item.content.split("\n")[0]
    assert (expected in line) if expected else ("older history exists" not in line)


@pytest.mark.parametrize("reach,clause", [
    (("fetch_thread_messages",), " — use fetch_thread_messages to read it"),
    (("search_slack",), ""),
    ((), ""),
])
def test_the_orphan_marker_names_its_tool_only_when_exposed(reach, clause):
    """T52's marker half, and the mutation this case exists for: a HARD-CODED marker literal
    passes every horizon case above and fails here, because history tools are a global switch
    and with them off the marker would instruct the model to call something it cannot."""
    absent = "1699000000.000000"
    pin = origin_pinned([msg(T1, text="reply", root=absent)], [], origin_root=None, reach=reach)
    marker = [i for i in serialize_stream(pin).items
              if i.content.startswith("[thread=")][0]
    assert marker.content == f"[thread={absent} began before this window{clause}]"


def test_the_horizon_carries_no_counts():
    """T53. RULING-12's append-stability property: adding a message changes NO byte of item 0,
    and every item before the new one is byte-identical across the two builds.

    r1 put the counts in item 0, which made the cache claim false — every new message changed the
    very top and invalidated the whole stream beneath it."""
    before = serialize_stream(pinned([msg(T0)]))
    after = serialize_stream(pinned([msg(T0), msg(T1)]))

    assert before.horizon_item.content == after.horizon_item.content
    for name in ("root_count", "message_count", "conversations", "messages)"):
        assert name not in before.horizon_item.content
    # THE PREFIX PROPERTY: the new message APPENDS — every prior item's bytes are unchanged.
    assert [i.content for i in before.items[:-1]] == [i.content for i in after.items[:-2]]


def test_no_message_text_is_ever_truncated():
    """T59. Full fidelity in BOTH blocks — the whole point of a shallow window is that what it
    does show is real."""
    long_text = "x" * 20_000
    stream = serialize_stream(origin_pinned([msg(T0, text=long_text)],
                                            [msg(T1, text=long_text)], origin_root=T1))
    assert long_text in stream.message_items[0].content
    assert long_text in stream.origin_items[0].content


def test_chrome_is_classified_once_and_validated_once(monkeypatch):
    """T66. ONE OPERATIONAL CLASSIFICATION plus ONE VALIDATION RECOMPUTATION — two different
    acts with two different purposes, which is why "validates" and "recomputes" are not in
    conflict.

    The pin STORES what selection already acted on; `__post_init__` recomputes over the DEDUPED
    UNION of both snapshots only to prove that value honest. Recompute-and-USE would silently
    discard it and reintroduce the divergence between selection and rendering that pinning the
    candidates exists to prevent.
    """
    chrome = msg(T2, text=":hourglass: Thinking...", sender="B0", sender_type="self")

    # ORIGIN-ONLY chrome: it never appears in the periphery, so validating over the periphery
    # alone would leave exactly this message unchecked.
    pin = origin_pinned([msg(T0)], [chrome], origin_root=T2)
    assert chrome.ts in pin.chrome_ts
    stream = serialize_stream(pin)
    assert stream.origin_items == (), "origin-only chrome must not render"

    # THE SERIALIZER READS THE MEMO — it never reclassifies. Swapping the classifier for a
    # recorder proves the rendered bytes are a function of the PINNED set.
    calls = []
    real = channel_stream.classify_chrome
    monkeypatch.setattr(channel_stream, "classify_chrome",
                        lambda messages, **kw: (calls.append([m.ts for m in messages]),
                                                real(messages, **kw))[1])
    serialize_stream(pin)
    assert calls == [], "serialization must not classify chrome at all"
    monkeypatch.undo()

    # ONE OPERATIONAL CLASSIFICATION PER DISTINCT TS: the validation recomputation sees the
    # deduped union, so a message present in BOTH snapshots is classified once, not twice.
    both = msg(T1, text=":hourglass: Thinking...", sender="B0", sender_type="self")
    seen = []
    monkeypatch.setattr(channel_stream, "classify_chrome",
                        lambda messages, **kw: (seen.append([m.ts for m in messages]),
                                                real(messages, **kw))[1])
    origin_pinned([msg(T0), both], [both], origin_root=T1)
    monkeypatch.undo()
    assert len(seen) == 1, "one recomputation, over the union"
    assert sorted(seen[0]) == sorted({T0, T1}), "the union is DEDUPED by ts"

    # A deliberately mismatched classification is a BUILD ERROR, not a silent correction.
    with pytest.raises(FreezeError):
        PinnedTuple(**{**pin.__dict__, "chrome_ts": frozenset({"9999.0"})})


@pytest.mark.parametrize("kind", ["own_in_flight", "own_chrome", "post_epoch_receiptless",
                                  "human"])
def test_a_below_floor_root_never_leaks_text(kind):
    """T72. `_root_snippet` reads NEITHER receipts NOR chrome, so a root that IS a fetched
    candidate but sits BELOW THE FLOOR would leak its text through the reply's header even
    though the root ITEM is correctly excluded — the own-output exclusion rule defeated by a
    header rather than by a bad predicate.

    Each shape is REAL, not relabelled: our own in-flight post carries an in_flight receipt, our
    chrome carries actual chrome text, the receiptless case sits above a real epoch, and the
    human case is a foreign message the floor alone excludes.
    """
    below, secret = "1699000000.000000", "SECRET-ROOT-TEXT"
    epoch = "1698000000.000000"          # REAL epoch: `below` is post-epoch, so no grandfather

    if kind == "own_in_flight":
        root = msg(below, text=secret, sender="B0", sender_type="self")
        cards = sidecars(epoch=epoch,
                         receipts=[ReceiptRec(ts=below, state="in_flight", turn_id="t",
                                              thread_root_ts=None)])
    elif kind == "own_chrome":
        root = msg(below, text=":hourglass: Thinking...", sender="B0", sender_type="self")
        cards = sidecars(epoch=epoch)
        secret = ":hourglass: Thinking..."
    elif kind == "post_epoch_receiptless":
        root = msg(below, text=secret, sender="B0", sender_type="self")
        cards = sidecars(epoch=epoch)      # post-epoch, NO receipt row -> excluded, loudly
    else:
        root = msg(below, text=secret, sender="U9", sender_type="human")
        cards = sidecars(epoch=epoch)

    reply = msg(T1, text="the reply", root=below)
    stream = serialize_stream(pinned([root, reply], cards=cards))

    assert [i.metadata["ts"] for i in stream.message_items] == [T1], kind
    header = stream.message_items[0].content.split("\n")[0]
    assert header.endswith(f"thread={below}]"), kind
    assert secret not in header, f"{kind}: a below-floor root must not leak its text"
    assert '~"' not in header, kind


def test_a_below_floor_root_needs_no_eligibility_lookup():
    """T72's other half: the decision is POSITIONAL, not evidential. A root outside the rendered
    window is skipped because it is outside — the pin is never asked about it, which is what
    makes the rule hold for a root the pin carries no row for at all."""
    below = "1699000000.000000"
    stream = serialize_stream(pinned([msg(T1, text="reply", root=below)]))
    header = stream.message_items[0].content.split("\n")[0]
    assert header.endswith(f"thread={below}]")
    # The root is not in the snapshot at all — no receipt, no chrome entry, nothing to look up.
    assert stream.pinned.sidecars.receipt_for(below) is None

    # A reply whose root IS rendered still gets its snippet: an honest referent when the root is
    # right there.
    root = msg(T0, text="the visible root text")
    present = serialize_stream(pinned([root, msg(T1, text="reply", root=T0)]))
    assert '~"' in present.message_items[1].content.split("\n")[0]


# ============================== the split-phase API: one shared periphery, N origin pins

def _shared_pin(periphery, *, cards=None, actor_map=(("U1", "alice"),), attempted=frozenset(),
                remaining=25, reach=("search_slack",)):
    """A `SharedChannelPin` over a fixed periphery — the object two origins are built FROM."""
    cards = cards if cards is not None else sidecars()
    config = serializer_config_snapshot()
    return SharedChannelPin(
        team_id=TEAM, channel_id=CH, h=T2, deadline_at=1000.0, serializer_config=config,
        generation=0, floor_read=cards.window[0], periphery_floor_ts=cards.window[0],
        reselected=False, periphery_candidates=tuple(periphery), periphery=tuple(periphery),
        periphery_sidecars=cards,
        periphery_chrome_ts=classify_chrome(tuple(periphery),
                                            chrome_markers=config["chrome_markers"]),
        actor_map=tuple(actor_map), actor_ids_attempted=frozenset(attempted),
        actor_lookups_remaining=remaining, coverage=COVERAGE, selection_version=1,
        reach_tools=tuple(reach), capability_profile_hash="caphash",
        tool_schema_version="tools-1", pages=SharedPageCounts(history=1, reply=0))


class _PinDB:
    """READ 2b only. The shared pin already carries READ 2a's rows."""

    def __init__(self):
        self.calls = []

    async def read_channel_sidecars_for_async(self, team_id, channel_id, message_ts):
        self.calls.append(list(message_ts))
        return {"ids": sorted(message_ts), "receipt_feature_epoch_ts": None, "receipts": [],
                "image_analyses": [], "document_extractions": [], "ambient_artifacts": [],
                "tool_usage": {}, "versions_hash": "originhash"}


async def test_two_origins_render_identical_pre_breakpoint_bytes():
    """T16. THE CROWN JEWEL. One shared periphery pin, TWO origin pins built from it ⇒
    byte-identical canonical items and equal `stream_sha256`, while `union_sha256` DIFFERS.

    This is what the whole stable-prefix layout exists to buy: every thread in the channel sends
    the same prefix bytes, so they share one cache entry.
    """
    cards = sidecars(documents=[{"message_ts": T0, "filename": "brief.pdf", "file_id": "F1",
                                 "summary": "the original summary"}])
    shared = _shared_pin([msg(T0, text="room chatter")], cards=cards)
    a_root, b_root = "1699500000.000000", "1699600000.000000"
    fetch_a = OriginFetch(origin_root_ts=a_root, messages=(msg(a_root, text="thread A"),),
                          pages=1, empty_fallback=False, deadline_at=1000.0)
    fetch_b = OriginFetch(origin_root_ts=b_root, messages=(msg(b_root, text="thread B"),),
                          pages=1, empty_fallback=False, deadline_at=1000.0)

    pin_a, _ = await build_origin_pin(shared, fetch_a, db=_PinDB())

    # THE DEEP FREEZE, PROBED **BETWEEN** THE TWO BUILDS — which is the only place it matters.
    # `frozen=True` protects the ATTRIBUTE, not what it POINTS AT, so code running here could
    # mutate a container and fork the very prefix these two calls exist to prove identical.
    # Each probe mutates something REAL: a marker inside the frozen config, a candidate list,
    # and — the row-level case a shallow freeze passes — a FIELD INSIDE an actual sidecar row.
    assert shared.periphery_sidecars.document_extractions, "the row probe needs a real row"
    for mutate in (lambda: shared.serializer_config["chrome_markers"].append("x"),
                   lambda: shared.periphery_candidates.append(msg(T1)),
                   lambda: shared.periphery_sidecars.document_extractions[0].__setitem__(
                       "summary", "rewritten between the two builds")):
        with pytest.raises((TypeError, AttributeError)):
            mutate()

    pin_b, _ = await build_origin_pin(shared, fetch_b, db=_PinDB())
    stream_a, stream_b = serialize_stream(pin_a), serialize_stream(pin_b)

    assert [i.content for i in stream_a.items] == [i.content for i in stream_b.items]
    assert stream_a.stream_sha256 == stream_b.stream_sha256
    assert stream_a.union_sha256 != stream_b.union_sha256, "two distinct origins must differ"


async def test_the_same_pin_serializes_identically_twice():
    """T17. Two Slack rebuilds are NOT equality proof — they differ in `H` by construction. One
    frozen pin serialized twice is."""
    shared = _shared_pin([msg(T0), msg(T1)])
    fetch = OriginFetch(origin_root_ts=T0, messages=(msg(T0),), pages=1, empty_fallback=False,
                        deadline_at=1000.0)
    pin, _ = await build_origin_pin(shared, fetch, db=_PinDB())

    first, second = serialize_stream(pin), serialize_stream(pin)
    assert first.stream_sha256 == second.stream_sha256
    assert first.union_sha256 == second.union_sha256
    assert first.receipts_membership_hash == second.receipts_membership_hash


class _RealResolverClient(SlackUtilitiesMixin):
    """A client carrying the REAL `resolve_usernames`, driven by a deterministic `users.info`.

    `fail_for` names ids whose lookup FAILS — which is the case the ownership rule turns on: an
    id the periphery ATTEMPTED and failed must never be retried by the origin phase, or one
    block's rendering of an actor would depend on which origin was built.
    """

    bot_handle = "chatgpt-dev"

    def __init__(self, fail_for=()):
        super().__init__()
        self.user_cache = {}
        self.db = None
        self.fail_for = set(fail_for)
        self.lookups = []
        self.app = SimpleNamespace(client=SimpleNamespace(users_info=self._users_info))

    def log_debug(self, *a, **k):
        pass

    async def _users_info(self, user):
        self.lookups.append(user)
        if user in self.fail_for:
            return {"ok": False}
        return {"ok": True, "user": {"name": f"name-{user}",
                                     "profile": {"display_name": f"name-{user}"}}}


async def test_origin_actors_cannot_starve_periphery_name_resolution():
    """T18. THE OWNERSHIP RULING, against the REAL resolver at its real cap.

    `resolve_usernames` resolves at most 25 ids per call and OMITS the rest, which then render
    as raw Slack ids. The landed actor map walks messages in order, so if origin actors entered
    that list first, an origin thread with more than 25 distinct authors would consume the whole
    budget and leave PERIPHERY actors rendering as raw ids — the periphery's bytes depending on
    which thread the turn originated in.

    Ordering alone does not fix it: the resolver serves and MUTATES an in-memory cache, so two
    origin pins run two different resolutions and can render two different periphery names for
    one id. So the SHARED phase resolves the periphery once and freezes it.
    """
    # (a) THE PERIPHERY MAP IS BUILT ONCE, and ids it ATTEMPTED — including one that FAILED —
    #     are never retried by the origin phase.
    failed_id = "U_FAILS"
    client = _RealResolverClient(fail_for={failed_id})
    periphery = [msg(T0, sender="U1"), msg(T1, sender=failed_id)]
    stats = {"budget": ACTOR_REMOTE_LOOKUP_DEFAULT, "remote_lookups": 0, "attempted_ids": set()}
    periphery_map = await _build_actor_map(client, periphery, stats=stats)

    assert dict(periphery_map).get("U1") == "name-U1"
    assert failed_id not in dict(periphery_map), "a failed lookup renders raw"
    assert failed_id in stats["attempted_ids"], "ATTEMPTED, not merely succeeded"

    remaining = max(0, stats["budget"] - stats["remote_lookups"])
    shared = _shared_pin(periphery, actor_map=periphery_map,
                         attempted=frozenset(stats["attempted_ids"]), remaining=remaining)

    # (b) THE ORIGIN GETS THE REMAINDER, floored at zero, and the excess renders as raw ids.
    #     THE ORIGIN ALSO CARRIES BOTH PERIPHERY ACTORS — including the one whose lookup FAILED.
    #     Without them here `skip_ids` would be untested: an id that never appears in the origin
    #     messages is not looked up whatever the skip set says.
    origin = ((msg("1699400000.000000", sender="U1"),
               msg("1699400001.000000", sender=failed_id))
              + tuple(msg(f"16995000{i:02d}.000000", sender=f"U{200 + i}") for i in range(40)))
    fetch = OriginFetch(origin_root_ts=origin[0].ts, messages=origin, pages=1,
                        empty_fallback=False, deadline_at=1000.0)
    before = len(client.lookups)
    pin, _ = await build_origin_pin(shared, fetch, db=_PinDB(), client=client)
    origin_lookups = client.lookups[before:]

    assert failed_id not in origin_lookups, "an id the periphery attempted is never retried"
    assert "U1" not in origin_lookups
    assert len(origin_lookups) <= remaining, "the origin spends only what is left"
    resolved_origin = [uid for uid in dict(pin.actor_map) if uid.startswith("U2")]
    assert len(resolved_origin) < 40, "authors past the remaining budget render as raw ids"
    # The periphery's rendered name is the SHARED map's, untouched.
    assert dict(pin.actor_map)["U1"] == "name-U1"

    # (c) TWO ORIGIN PINS FROM ONE SHARED PIN render byte-identical periphery names, EVEN THOUGH
    #     the first origin's resolution mutated the client's name cache. This is the case
    #     ordering alone could not cover.
    other = tuple(msg(f"16996000{i:02d}.000000", sender=f"U{300 + i}") for i in range(40))
    fetch_b = OriginFetch(origin_root_ts=other[0].ts, messages=other, pages=1,
                          empty_fallback=False, deadline_at=1000.0)
    pin_b, _ = await build_origin_pin(shared, fetch_b, db=_PinDB(), client=client)

    stream_a, stream_b = serialize_stream(pin), serialize_stream(pin_b)
    assert stream_a.stream_sha256 == stream_b.stream_sha256
    assert [i.content for i in stream_a.items] == [i.content for i in stream_b.items]


async def test_an_empty_periphery_still_reports_its_untouched_budget():
    """T18's empty-periphery case. A periphery with NO human actors never reaches
    `resolve_usernames` at all (the landed `if human_ids:` guard), yet the shared pin must still
    construct — with `attempted_ids` empty, zero lookups spent, and the FULL cap left for the
    origin phase.

    THE MUTATION THIS CATCHES: leaving `stats` for the resolver to fill raises `KeyError` on
    exactly this legal path, because nothing ever fills it.
    """
    client = _RealResolverClient()
    stats = {"budget": ACTOR_REMOTE_LOOKUP_DEFAULT, "remote_lookups": 0, "attempted_ids": set()}
    actor_map = await _build_actor_map(client, [], stats=stats)

    assert actor_map == ()
    assert client.lookups == [], "no human ids means no resolver call at all"
    remaining = max(0, stats["budget"] - stats["remote_lookups"])
    assert remaining == ACTOR_REMOTE_LOOKUP_DEFAULT

    shared = _shared_pin([], attempted=frozenset(stats["attempted_ids"]), remaining=remaining)
    assert shared.actor_ids_attempted == frozenset()
    assert shared.actor_lookups_remaining == ACTOR_REMOTE_LOOKUP_DEFAULT


async def test_a_duplicated_message_is_counted_once_in_receipt_membership():
    """§8's UNIQUE-ts accounting. RULING-1 renders an origin message that also sits in the
    periphery TWICE, deliberately — and a receipt is a fact about a MESSAGE, not about how many
    times it appeared, so the duplication must not reach the counts or the hash.

    Mutation-check: appending per RENDER without canonicalizing double-counts exactly the
    message the duplication is about, and the membership hash moves with a rendering decision
    rather than with the receipt state it claims to describe.
    """
    ts = "1700000500.000000"
    own = msg(ts, text="our own finalized post", sender="U_BOT", sender_type="self")
    cards = sidecars(receipts=[ReceiptRec(ts=ts, state="finalized", turn_id="t1",
                                          thread_root_ts=None)])
    shared = _shared_pin([own], cards=cards)
    fetch = OriginFetch(origin_root_ts=ts, messages=(own,), pages=1, empty_fallback=False,
                        deadline_at=1000.0)
    pin, _ = await build_origin_pin(shared, fetch, db=_PinDB())
    stream = serialize_stream(pin)

    # It really is rendered twice — once in the room, once in the thread.
    assert any(item.metadata.get("ts") == ts for item in stream.message_items)
    assert any(item.metadata.get("ts") == ts for item in stream.origin_items)
    # …and counted ONCE.
    assert stream.receipts_included == (ts,)
    assert stream.receipts_excluded == ()
    # …including in the telemetry field §8 actually governs, which is derived from that tuple.
    fields = stream.stream_render_fields()
    assert fields["receipts_included_count"] == 1
    assert fields["receipts_excluded_count"] == 0

    # The hash is over the DISTINCT membership, so it matches a build where the same message was
    # rendered only once. The duplication cannot move it.
    periphery_only, _ = await build_origin_pin(
        _shared_pin([own], cards=cards),
        OriginFetch(origin_root_ts=None, messages=(), pages=0, empty_fallback=False,
                    deadline_at=1000.0),
        db=_PinDB())
    assert serialize_stream(periphery_only).receipts_membership_hash == (
        stream.receipts_membership_hash)


class _NamingClient:
    """A client whose resolver would happily name anything it is asked about."""

    bot_handle = "chatgpt-dev"
    app = None

    async def resolve_usernames(self, ids, api_client, max_remote_lookups=25, stats=None):
        if stats is not None:
            stats["remote_lookups"] = len(ids)
            stats.setdefault("attempted_ids", set()).update(ids)
        return {uid: f"name-{uid}" for uid in ids}


async def test_an_origin_bot_name_cannot_rename_a_periphery_bot():
    """T18's BOT case, and it is the one that was actually broken.

    `skip_ids` guarded only the branch that spends remote lookups, so a bot appearing in the
    PERIPHERY with no `raw_bot_name` but in the ORIGIN with one had that name added by the origin
    phase — and `render_header` then rendered it into the periphery. Two origins, two different
    pre-breakpoint prefixes: the single thing the stable prefix may never do.

    It passed the live probe only because every bot in the probed channel carried a raw name. The
    invariant cannot rest on that.

    Mutation-check: restoring the human-only `skip_ids` guard renames the periphery bot from the
    origin and forks `stream_sha256` between the two pins below.
    """
    bot_id = "B_OTHER"
    # THE SAME id. In the room it arrives with no usable name; inside the thread Slack gave one.
    nameless = msg(T0, text="deploy finished", sender=bot_id, sender_type="other_bot")
    named = msg("1699500000.000000", text="deploy finished", sender=bot_id,
                sender_type="other_bot", bot_name="Deploybot")

    shared = _shared_pin([nameless], actor_map=(), attempted=frozenset())
    fetch_a = OriginFetch(origin_root_ts=named.ts, messages=(named,), pages=1,
                          empty_fallback=False, deadline_at=1000.0)
    fetch_b = OriginFetch(origin_root_ts="1699600000.000000",
                          messages=(msg("1699600000.000000", text="unrelated thread"),),
                          pages=1, empty_fallback=False, deadline_at=1000.0)

    pin_a, _ = await build_origin_pin(shared, fetch_a, db=_PinDB(), client=_NamingClient())
    pin_b, _ = await build_origin_pin(shared, fetch_b, db=_PinDB(), client=_NamingClient())
    stream_a, stream_b = serialize_stream(pin_a), serialize_stream(pin_b)

    # THE PERIPHERY BYTES ARE IDENTICAL, though only one origin knew the bot's name.
    assert [i.content for i in stream_a.items] == [i.content for i in stream_b.items]
    assert stream_a.stream_sha256 == stream_b.stream_sha256
    # …and the periphery renders the bot as the ROOM saw it: unnamed.
    periphery_header = stream_a.message_items[0].content.split("\n")[0]
    assert "Deploybot" not in periphery_header
