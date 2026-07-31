"""Grammar v2 of the channel stream, frozen byte for byte.

This is the contract the prompt cache, the stale-send guard and every cross-thread awareness
claim rest on. `serialize_stream` is a pure function of the pinned tuple, so the tests are
literal expected strings rather than shape assertions: if a header, a marker or an escape moves,
that is a serializer version change and it has to be a deliberate one.

The MESSAGE-item grammar below is v1's, unchanged, and it stays asserted here: v2 changed which
items exist and what the framing items say, not how a message renders. The v2-only surface —
the summary block, the anchor map, the escaping and the freeze — lives in
`test_channel_stream_v2.py`.
"""
from __future__ import annotations

import hashlib

import pytest

from message_processor.channel_stream import (
    DELETED_SNIPPET,
    END_MARKER_TEXT,
    HORIZON_TEMPLATE,
    SERIALIZER_VERSION,
    ChannelStream,
    CoveragePin,
    PinnedTuple,
    ReceiptRec,
    SidecarPin,
    StreamTimestampError,
    _dedup,
    iso_minute,
    serialize_stream,
    serializer_config_snapshot,
)
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
COVERAGE = CoveragePin(start_ts="1699999940.000000", status="complete", reason="genesis")
HORIZON = HORIZON_TEMPLATE.format(
    summary_clause="no summary", coverage_start_ts=COVERAGE.start_ts,
    reason_clause="genesis: the channel's first message")


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
    return PinnedTuple(
        team_id=TEAM, channel_id=CH, snapshot=None, window=cards.window, H=h,
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


def test_horizon_names_the_coverage_reason_and_falls_back_when_absent():
    cards = sidecars()
    pin = pinned([msg(T0)], cards=cards)
    object.__setattr__(pin, "coverage", CoveragePin(COVERAGE.start_ts, "limited", None))
    assert "(unknown)" in serialize_stream(pin).horizon_item.content


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
    reply = msg(T1, text="orphan", root="1699000000.000000")
    header = contents(serialize_stream(pinned([reply])))[1].split("\n")[0]
    assert header.endswith("thread=1699000000.000000]")


def test_flags_render_in_pinned_order():
    m = msg(T1, text="x", root=T0, edited_ts="1700000070.000000", broadcast=True)
    header = contents(serialize_stream(pinned([m])))[1].split("\n")[0]
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
    assert stream.receipts_included == (narrow, wide)  # snapshot order is preserved
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
    root = msg(T0, text="root")
    reply = msg(T1, text="reply", root=T0, origin=ORIGIN_REPLIES)
    other = msg(T2, text="elsewhere")
    stream = serialize_stream(pinned([root, reply, other]))
    assert [i.metadata["ts"] for i in stream.origin_slice(T0)] == [T1]
    assert stream.trigger_view(T2).metadata["ts"] == T2
    assert stream.trigger_view("1600000000.000000") is None
    assert stream.origin_slice(None) == ()


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
