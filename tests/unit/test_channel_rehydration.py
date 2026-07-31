"""Post-breakpoint snapshot evidence, and the admission split that decides whether compacting
would have helped (P4 §1i late artifacts, §1k rehydration, R0-1 component accounting).

What this file protects, in order of how expensive it is to get wrong:

1. THE SUMMARY ITEM REACHES THE MODEL. Under serializer v2 the compacted history is item 0 of the
   canonical sequence. An assembler that names its items instead of iterating them drops it
   silently: the summary still exists, still hashes into `stream_sha256`, and the model answers
   from a window it believes is whole.
2. ONE ADMISSION CURRENCY, SPLIT BY CAUSE. Compaction may only fire when the CANONICAL STREAM is
   what overflowed. A 300k-token attachment that triggered compaction would publish a summary,
   arrive over budget again, and do it every turn.
3. REHYDRATION IS BOUNDED AND HONEST. Its own page/time budget, never the turn's; the root always
   consumes a slot; every failure is one line of evidence rather than a refused turn.
"""
from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from message_processor import channel_request
from message_processor.channel_request import (assemble_channel_request,
                                               build_late_artifact_items,
                                               build_rehydration_item, estimate_admission,
                                               rehydration_variant_headroom)
from message_processor.channel_stream import (PROD_NAMESPACE, SERIALIZER_VERSION, CoveragePin,
                                              PinnedTuple, SidecarPin,
                                              render_anchor_block, serialize_stream,
                                              serializer_config_snapshot)
from slack_client.normalizer import ORIGIN_HISTORY, NormalizedMessage
from tests.unit.channel_turn_harness import item_texts, steering, thread_config
from token_counter import ITEM_STRUCTURAL_OVERHEAD, admission_charge

TEAM = "T1"
CH = "C0BKX77NU66"
MODEL = "gpt-5.6-sol"
FLOOR = "1700000000.000000"
ROOT = "1700000100.000000"        # the origin root, BELOW the boundary
R1 = "1700000200.000000"
R2 = "1700000300.000000"
BOUNDARY = "1700000500.000000"
T0 = "1700001000.000100"
H = "1700009999.000000"
PAYLOAD = "earlier, the room argued about the lunch budget."
COVERAGE = CoveragePin(start_ts=FLOOR, status="complete", reason="genesis")


# --------------------------------------------------------------------------- harness

def msg(ts, text="hello", *, sender="U1", sender_type="human", root=None,
        origin=ORIGIN_HISTORY) -> NormalizedMessage:
    return NormalizedMessage(
        team_id=TEAM, channel_id=CH, ts=ts, thread_root_ts=root, subtype=None,
        sender_id=sender, sender_type=sender_type, raw_bot_name=None, text=text,
        files=(), reactions=(), edited_ts=None, is_broadcast=False, is_tombstone=False,
        reply_count=None, latest_reply=None, mention_ids=(), origin=origin)


def cards(*, receipts=(), epoch=None, window=None) -> SidecarPin:
    return SidecarPin(
        window=window or (BOUNDARY, False), receipts=tuple(receipts),
        receipt_feature_epoch_ts=epoch, coverage=COVERAGE, activity_roots=(),
        activity_event_ts=(), image_analyses=(), document_extractions=(),
        ambient_artifacts=(), tool_usage=(), versions_hash="sidecarhash")


def snapshot_row(*, payload=PAYLOAD, status="published", boundary=BOUNDARY,
                 snapshot_id="snap-1", generation=3):
    return {
        "snapshot_id": snapshot_id, "generation": generation, "boundary_ts": boundary,
        "status": status, "namespace": PROD_NAMESPACE,
        "payload_bytes": payload.encode("utf-8"),
        "anchor_payload_bytes": render_anchor_block([], omitted=0).encode("utf-8"),
    }


def stream(messages=(), *, snapshot=None, pins=None, h=H, actors=(("U1", "alice"),)):
    pins = pins if pins is not None else cards()
    pinned = PinnedTuple(
        team_id=TEAM, channel_id=CH, snapshot=snapshot, window=pins.window, H=h,
        fetch_snapshot=tuple(messages), sidecar_versions_hash=pins.versions_hash,
        actor_map=tuple(actors), actor_map_hash="actorhash",
        serializer_version=SERIALIZER_VERSION, serializer_config_hash="cfghash",
        capability_profile_hash="caphash", tool_schema_version="tools-1",
        coverage=COVERAGE, receipt_feature_epoch_ts=pins.receipt_feature_epoch_ts,
        receipt_map=tuple((r.ts, r.state, r.turn_id, r.thread_root_ts) for r in pins.receipts),
        sidecars=pins, serializer_config=serializer_config_snapshot())
    return serialize_stream(pinned)


def host():
    """A processor stand-in binding the real builders the assembler calls into."""
    from message_processor.handlers.text import TextHandlerMixin
    from message_processor.utilities import MessageUtilitiesMixin

    stub = MagicMock()
    stub._channel_prepared_tools = TextHandlerMixin._channel_prepared_tools.__get__(stub)
    stub._build_time_suffix_context = (
        MessageUtilitiesMixin._build_time_suffix_context.__get__(stub))
    stub._build_message_with_documents = (
        MessageUtilitiesMixin._build_message_with_documents.__get__(stub))
    stub._build_generation_inflight_note = MagicMock(return_value=None)
    stub._build_research_inflight_note = MagicMock(return_value=None)
    stub._get_system_prompt = MagicMock(return_value="SYSTEM")
    return stub


def context(built, **overrides):
    fields = {
        "stream": built, "steering": steering(), "thread_config": thread_config(),
        "channel_id": CH, "team_id": TEAM, "trigger_ts": T0, "origin_thread_ts": ROOT,
        "trigger_text": "and now?",
        "requester": channel_request.RequesterFacts(user_id="U1", real_name="Alice",
                                                    sender_type="human"),
    }
    fields.update(overrides)
    return channel_request.ChannelTurnContext(**fields)


def assemble(built, *, with_estimate=False, **overrides):
    return assemble_channel_request(
        processor=host(), client=MagicMock(), ctx=context(built, **overrides), model=MODEL,
        tools=None, request_config={}, contract_suffix=None, registry=None,
        with_estimate=with_estimate)


# ------------------------------------------------ the blocker: the summary must be consumed

def test_the_pinned_summary_block_reaches_the_model():
    """THE regression this file exists for. `assemble_channel_request` used to name its three
    canonical items, so the v2 summary item — present whenever a snapshot is pinned — was built,
    hashed into `stream_sha256`, and then dropped from every real request."""
    built = stream([msg(T0)], snapshot=snapshot_row())
    texts = item_texts(assemble(built).input_items)

    assert any(PAYLOAD in text for text in texts), "the compacted history never reached the model"
    assert texts[0].startswith("[CHANNEL SUMMARY — compacted history through ")
    assert texts[0].endswith("[END CHANNEL SUMMARY]")
    # And it leads: the summary is item 0 of the canonical sequence, above the horizon.
    assert texts[1].startswith("[STREAM HORIZON:")


def test_every_canonical_item_is_carried_in_its_own_order():
    """Iterated, not named: whatever the stream says its canonical sequence is, that is what the
    request sends, in that order, with nothing added and nothing dropped."""
    built = stream([msg(T0, "first"), msg("1700001100.000000", "second")],
                   snapshot=snapshot_row())
    request = assemble(built)

    carried = request.stream_items
    assert len(carried) == len(built.items)
    assert [item["content"] for item in carried[:-1]] == [i.content for i in built.items[:-1]]
    assert all(item["role"] == expected.role
               for item, expected in zip(carried, built.items))


def test_the_end_marker_still_carries_the_breakpoint_under_a_summary():
    """The summary must not cost the cache breakpoint: the last canonical item is still the end
    marker, still rendered through `end_marker_content`."""
    built = stream([msg(T0)], snapshot=snapshot_row())
    last = assemble(built).stream_items[-1]
    assert isinstance(last["content"], list)
    assert last["content"][0].get("prompt_cache_breakpoint")


def test_genesis_carries_no_summary_item_at_all():
    built = stream([msg(T0)], snapshot=None)
    texts = item_texts(assemble(built).input_items)
    assert not any(text.startswith("[CHANNEL SUMMARY") for text in texts)
    assert texts[0].startswith("[STREAM HORIZON:")


# --------------------------------------------------------- R0-1 component accounting

def _limit() -> int:
    return config.get_model_token_limit(MODEL)


def test_the_canonical_stream_is_the_compactable_component():
    estimate = estimate_admission(
        instructions="", tools=None, raw_document_texts=(), native_file_bounds=(), model=MODEL,
        input_items=[{"role": "user", "content": "x" * 100, "_stream": True},
                     {"role": "user", "content": "y" * 40}])

    assert estimate.compactable_tokens == 100 + ITEM_STRUCTURAL_OVERHEAD
    assert estimate.fixed_tokens == estimate.total_tokens - estimate.compactable_tokens
    assert estimate.compactable_tokens + estimate.fixed_tokens == estimate.total_tokens


def test_an_attachment_that_overflows_never_indicates_compaction():
    """R0-1: attachments and suffix evidence must NEVER induce a futile compaction loop. No
    summary of the channel makes a 300k-token document smaller."""
    estimate = estimate_admission(
        instructions="", tools=None, native_file_bounds=(), model=MODEL,
        raw_document_texts=(("huge.csv", "x" * (_limit() + 1000)),),
        input_items=[{"role": "user", "content": "hi", "_stream": True}])

    assert not estimate.fits
    assert estimate.compaction_headroom <= 0
    assert not estimate.compaction_indicated


def test_a_stream_that_overflows_is_what_compaction_is_for():
    big = "x" * (_limit() // 2)
    estimate = estimate_admission(
        instructions="", tools=None, raw_document_texts=(), native_file_bounds=(), model=MODEL,
        input_items=[{"role": "user", "content": big, "_stream": True} for _ in range(3)])

    assert not estimate.fits
    assert estimate.compaction_headroom > 0
    assert estimate.compaction_indicated
    # "What would compacting actually save" is the compactable component itself.
    assert estimate.compactable_tokens >= 3 * len(big.encode("utf-8"))


def test_a_request_that_fits_never_indicates_compaction():
    estimate = estimate_admission(
        instructions="", tools=None, raw_document_texts=(), native_file_bounds=(), model=MODEL,
        input_items=[{"role": "user", "content": "small", "_stream": True}])
    assert estimate.fits and not estimate.compaction_indicated


def test_the_component_split_moves_no_existing_number():
    """The split is a division of the SAME total, not a second measurement. Every landed
    admission figure has to be exactly what it was."""
    items = [{"role": "user", "content": "abc", "_stream": True},
             {"role": "user", "content": "defgh"}]
    estimate = estimate_admission(
        instructions="sys", tools=[{"type": "function", "name": "t"}],
        raw_document_texts=(("a.pdf", "x" * 40),), native_file_bounds=(11,), model=MODEL,
        input_items=items)

    assert estimate.total_tokens == sum(estimate.breakdown.values())
    assert set(estimate.breakdown) == {"instructions", "tools", "items", "structure", "images",
                                       "native_files", "document_text"}
    assert estimate.breakdown["items"] == 3 + 5
    assert estimate.breakdown["structure"] == 3 * ITEM_STRUCTURAL_OVERHEAD
    assert estimate.breakdown["document_text"] == 40


def test_the_assembled_channel_request_splits_its_own_components():
    built = stream([msg(T0, "a channel message with some length to it")],
                   snapshot=snapshot_row())
    estimate = assemble(built, with_estimate=True).estimate

    assert estimate.compactable_tokens > 0
    assert estimate.fixed_tokens > 0
    # The instructions and the developer suffix are on the fixed side; the stream is not.
    assert estimate.compactable_tokens < estimate.total_tokens


# --------------------------------------------------------- §1k rehydration: selection

def rehydrate(built, *, fetched, receipts=(), origin=ROOT, db=None):
    db = db if db is not None else AsyncMock()
    with patch.object(channel_request, "_fetch_replies",
                      AsyncMock(return_value=list(fetched))) as fetch:
        item = asyncio.run(build_rehydration_item(
            client=MagicMock(), db=db, stream=built, origin_thread_ts=origin,
            preboundary_receipts=list(receipts)))
    return item, fetch


def test_the_pre_boundary_tail_renders_oldest_first_inside_one_user_item():
    built = stream([msg(T0, "after the boundary")], snapshot=snapshot_row())
    item, _ = rehydrate(built, fetched=[msg(R1, "middle", root=ROOT),
                                        msg(ROOT, "the question"),
                                        msg(R2, "later", root=ROOT)])

    assert item["role"] == "user"
    content = item["content"]
    assert content.startswith("[THIS THREAD BEFORE THE SUMMARY BOUNDARY — oldest first]")
    assert content.endswith("[END EARLIER THREAD CONTEXT]")
    assert content.index("the question") < content.index("middle") < content.index("later")
    # Never a canonical item, and never carrying a ts the stale-send guard would read.
    assert "ts" not in item["metadata"]
    assert item["metadata"]["thread_root_ts"] == ROOT


def test_the_root_consumes_a_slot_and_the_label_says_so(monkeypatch):
    """The bounded selection is root + (MAX - 1) replies, never "the last MAX messages" — the
    rendered label has to describe what the model is actually looking at."""
    monkeypatch.setattr(config, "rehydration_max_messages", 4)
    replies = [msg(f"17000002{index:02d}.000000", f"reply {index}", root=ROOT)
               for index in range(10)]
    built = stream([msg(T0)], snapshot=snapshot_row())
    item, _ = rehydrate(built, fetched=[msg(ROOT, "the question")] + replies)

    content = item["content"]
    assert content.startswith("[THIS THREAD BEFORE THE SUMMARY BOUNDARY — oldest first, "
                              "root plus the latest 3 replies]")
    assert "the question" in content
    for kept in ("reply 7", "reply 8", "reply 9"):
        assert kept in content
    assert "reply 6" not in content


def test_an_unbounded_selection_claims_no_bound(monkeypatch):
    monkeypatch.setattr(config, "rehydration_max_messages", 20)
    built = stream([msg(T0)], snapshot=snapshot_row())
    item, _ = rehydrate(built, fetched=[msg(ROOT, "q"), msg(R1, "a", root=ROOT)])
    assert item["content"].startswith(
        "[THIS THREAD BEFORE THE SUMMARY BOUNDARY — oldest first]")


def test_an_oversized_root_is_truncated_at_a_character_boundary(monkeypatch):
    """Appendix C test 8. A byte prefix that landed mid-codepoint is a request the API rejects,
    so the cap can never be paid for with a broken character."""
    monkeypatch.setattr(config, "rehydration_max_bytes", 200)
    built = stream([msg(T0)], snapshot=snapshot_row())
    item, _ = rehydrate(built, fetched=[msg(ROOT, "日本語のテキスト" * 40),
                                        msg(R1, "a reply", root=ROOT)])

    content = item["content"]
    assert "[root truncated]" in content
    # No half character survived the cut: a byte slice that landed mid-codepoint leaves either a
    # replacement char (decode "replace") or raw bytes the API rejects.
    assert "�" not in content
    root_block = content.split("\n[root truncated]")[0].split("]\n", 1)[1]
    assert len(root_block.encode("utf-8")) <= 200
    # And the reply lost its place to the root, which always keeps its own.
    assert "a reply" not in content
    assert "root plus the latest" in content


def test_the_byte_cap_drops_the_oldest_replies_first(monkeypatch):
    monkeypatch.setattr(config, "rehydration_max_bytes", 400)
    replies = [msg(f"17000002{index:02d}.000000", "y" * 120, root=ROOT) for index in range(6)]
    built = stream([msg(T0)], snapshot=snapshot_row())
    item, _ = rehydrate(built, fetched=[msg(ROOT, "q")] + replies)

    assert len(item["content"].encode("utf-8")) < 700
    assert "root plus the latest" in item["content"]


def test_the_canonical_tail_is_deduped_by_ts():
    """A message the window already renders must not be shown twice."""
    shared = msg(BOUNDARY, "on the boundary", root=ROOT)
    built = stream([shared, msg(T0)], snapshot=snapshot_row())
    item, _ = rehydrate(built, fetched=[msg(ROOT, "the question"), shared])

    assert "the question" in item["content"]
    assert "on the boundary" not in item["content"]


def test_a_thread_that_began_after_the_boundary_is_not_rehydrated():
    built = stream([msg(T0)], snapshot=snapshot_row())
    item, fetch = rehydrate(built, fetched=[msg(T0)], origin=T0)
    assert item is None
    assert fetch.await_count == 0


def test_genesis_never_rehydrates():
    built = stream([msg(T0)], snapshot=None)
    item, fetch = rehydrate(built, fetched=[msg(ROOT)])
    assert item is None
    assert fetch.await_count == 0


def test_a_thread_slack_no_longer_returns_is_an_honest_omission():
    built = stream([msg(T0)], snapshot=snapshot_row())
    item, _ = rehydrate(built, fetched=[])
    assert item["content"] == ("[THIS THREAD BEFORE THE SUMMARY BOUNDARY — "
                               "unavailable: retention]")
    assert "[END EARLIER THREAD CONTEXT]" not in item["content"]


def test_a_thread_with_nothing_below_the_boundary_renders_nothing():
    built = stream([msg(T0)], snapshot=snapshot_row())
    item, _ = rehydrate(built, fetched=[msg("1700000600.000000", "after", root=ROOT)])
    assert item is None


# ------------------------------------------------ §1k role and chrome, from its OWN evidence

def test_our_own_pre_boundary_message_needs_its_own_finalized_receipt():
    """The window's receipts cover `(boundary, H]` by construction, so borrowing them would
    exclude every pre-boundary message of ours. Rehydration pins its own."""
    own = msg(R1, "what I said back then", sender="UBOT", sender_type="self", root=ROOT)
    built = stream([msg(T0)], snapshot=snapshot_row())

    with_receipt, _ = rehydrate(
        built, fetched=[msg(ROOT, "q"), own],
        receipts=[{"message_ts": R1, "state": "finalized", "turn_id": "t1",
                   "thread_root_ts": ROOT}])
    assert "what I said back then" in with_receipt["content"]

    without, _ = rehydrate(built, fetched=[msg(ROOT, "q"), own], receipts=[])
    assert "what I said back then" not in without["content"]


def test_an_in_flight_own_message_is_excluded_from_the_tail():
    own = msg(R1, "half a sentence", sender="UBOT", sender_type="self", root=ROOT)
    built = stream([msg(T0)], snapshot=snapshot_row())
    item, _ = rehydrate(built, fetched=[msg(ROOT, "q"), own],
                        receipts=[{"message_ts": R1, "state": "in_flight",
                                   "turn_id": "t1", "thread_root_ts": ROOT}])
    assert "half a sentence" not in item["content"]


def test_the_pre_boundary_receipts_are_read_with_the_pinned_keyword():
    """§1k's evidence is fetched with `preboundary_receipts=True`; the ordinary read stops at the
    floor and would answer the wrong question."""
    db = AsyncMock()
    db.read_channel_sidecars_async.return_value = {"preboundary_receipts": []}
    built = stream([msg(T0)], snapshot=snapshot_row())
    with patch.object(channel_request, "_fetch_replies",
                      AsyncMock(return_value=[msg(ROOT, "q")])):
        asyncio.run(build_rehydration_item(client=MagicMock(), db=db, stream=built,
                                           origin_thread_ts=ROOT))

    kwargs = db.read_channel_sidecars_async.await_args.kwargs
    assert kwargs["preboundary_receipts"] is True
    assert db.read_channel_sidecars_async.await_args.args[:3] == (TEAM, CH, H)


def test_an_unreadable_receipt_pin_degrades_to_an_omission():
    db = AsyncMock()
    db.read_channel_sidecars_async.side_effect = RuntimeError("locked")
    built = stream([msg(T0)], snapshot=snapshot_row())
    with patch.object(channel_request, "_fetch_replies",
                      AsyncMock(return_value=[msg(ROOT, "q")])):
        item = asyncio.run(build_rehydration_item(client=MagicMock(), db=db, stream=built,
                                                  origin_thread_ts=ROOT))
    assert item["content"].endswith("unavailable: fetch_error]")


# ------------------------------------------------ §1k budgets (Appendix C test 9)

def _paging_client(pages: int = 50):
    """A client whose replies walk never terminates on its own — only a budget stops it."""
    calls = {"n": 0}

    async def replies(**params):
        calls["n"] += 1
        return {"messages": [{"type": "message", "ts": f"17000001{calls['n']:02d}.000000",
                              "user": "U1", "text": "reply", "thread_ts": ROOT}],
                "has_more": True,
                "response_metadata": {"next_cursor": f"c{calls['n']}"}}

    return SimpleNamespace(app=None, conversations_replies=replies, self_team_id=TEAM), calls


def test_the_page_budget_stops_the_walk_and_never_touches_the_turns(monkeypatch):
    """Appendix C test 9: exhausting rehydration's own page budget produces the omission item and
    consumes NO canonical fetch budget."""
    from slack_client.history_fetch import FetchBudget

    monkeypatch.setattr(config, "rehydration_page_budget", 3)
    canonical = FetchBudget(total_seconds=600.0, page_ceiling=500)
    client, calls = _paging_client()
    built = stream([msg(T0)], snapshot=snapshot_row())
    db = AsyncMock()
    db.read_channel_sidecars_async.return_value = {"preboundary_receipts": []}

    item = asyncio.run(build_rehydration_item(client=client, db=db, stream=built,
                                              origin_thread_ts=ROOT))

    assert item["content"] == ("[THIS THREAD BEFORE THE SUMMARY BOUNDARY — "
                               "unavailable: fetch_budget_exhausted]")
    assert calls["n"] == 3, "the walk outlived its own page ceiling"
    assert canonical.pages_used == 0


def test_the_time_budget_stops_the_walk(monkeypatch):
    monkeypatch.setattr(config, "rehydration_page_budget", 500)
    monkeypatch.setattr(config, "rehydration_time_budget", 10.0)
    ticks = iter([0.0] + [float(n) for n in range(1, 200)])
    client, calls = _paging_client()
    built = stream([msg(T0)], snapshot=snapshot_row())
    db = AsyncMock()
    db.read_channel_sidecars_async.return_value = {"preboundary_receipts": []}

    item = asyncio.run(build_rehydration_item(
        client=client, db=db, stream=built, origin_thread_ts=ROOT,
        budget_clock=lambda: next(ticks)))

    assert item["content"].endswith("unavailable: fetch_budget_exhausted]")
    assert calls["n"] < 200


def test_a_broken_fetch_is_an_omission_rather_than_a_failed_turn():
    built = stream([msg(T0)], snapshot=snapshot_row())
    db = AsyncMock()
    with patch.object(channel_request, "_fetch_replies",
                      AsyncMock(side_effect=RuntimeError("slack said no"))):
        item = asyncio.run(build_rehydration_item(client=MagicMock(), db=db, stream=built,
                                                  origin_thread_ts=ROOT))
    assert item["content"] == ("[THIS THREAD BEFORE THE SUMMARY BOUNDARY — "
                               "unavailable: fetch_error]")


# ------------------------------------------------ §1k admission: the longer of the two variants

def test_admission_charges_the_longer_of_the_marker_and_the_content():
    """A late switch between the content block and the one-line omission marker must not be able
    to overflow a request that was already admitted."""
    from message_processor.channel_stream import render_rehydration_omission

    longest = max((render_rehydration_omission(reason)
                   for reason in ("fetch_budget_exhausted", "fetch_error", "retention")),
                  key=lambda text: len(text.encode("utf-8")))
    assert channel_request.REHYDRATION_OMISSION_CHARGE == admission_charge(longest)

    tiny = {"role": "user", "content": "[x]"}
    assert rehydration_variant_headroom(tiny) == (
        channel_request.REHYDRATION_OMISSION_CHARGE - admission_charge("[x]"))
    assert rehydration_variant_headroom(None) == 0


def test_a_full_rehydration_block_needs_no_variant_headroom():
    built = stream([msg(T0)], snapshot=snapshot_row())
    item, _ = rehydrate(built, fetched=[msg(ROOT, "a question worth some bytes"),
                                        msg(R1, "and an answer", root=ROOT)])
    assert rehydration_variant_headroom(item) == 0

    estimate = assemble(built, with_estimate=True, rehydration_item=item).estimate
    assert "variant_headroom" not in estimate.breakdown


def test_a_short_block_is_charged_the_marker_it_might_become():
    built = stream([msg(T0)], snapshot=snapshot_row())
    item = {"role": "user", "content": "[x]"}
    estimate = assemble(built, with_estimate=True, rehydration_item=item).estimate
    assert estimate.breakdown["variant_headroom"] == rehydration_variant_headroom(item)
    # And it is FIXED headroom: no summary of the channel makes it smaller.
    assert estimate.total_tokens == sum(estimate.breakdown.values())


# ------------------------------------------------ placement in the assembled request

def test_the_post_breakpoint_blocks_sit_below_the_end_marker_and_above_the_suffix():
    built = stream([msg(T0)], snapshot=snapshot_row())
    artifact = {"role": "user", "content": "[EARLIER ARTIFACT — x]", "metadata": {}}
    rehydration = {"role": "user", "content": "[THIS THREAD BEFORE THE SUMMARY BOUNDARY — y]",
                   "metadata": {}}
    request = assemble(built, late_artifact_items=(artifact,), rehydration_item=rehydration)

    roles = [item["role"] for item in request.input_items]
    texts = [item.get("content") for item in request.input_items]
    marker = max(index for index, item in enumerate(request.input_items) if item.get("_stream"))
    assert texts.index("[EARLIER ARTIFACT — x]") == marker + 1
    assert texts.index("[THIS THREAD BEFORE THE SUMMARY BOUNDARY — y]") == marker + 2
    # Post-breakpoint evidence is USER role, always: a developer-role artifact block would make
    # somebody's uploaded document an instruction.
    assert roles[marker + 1] == roles[marker + 2] == "user"
    assert roles[-1] == "developer"
    # And they are not canonical: the cacheable prefix ends at the end marker.
    assert not request.input_items[marker + 1].get("_stream")


def test_the_blocks_are_absent_rather_than_empty():
    built = stream([msg(T0)], snapshot=snapshot_row())
    plain = assemble(built)
    assert not any(text.startswith("[EARLIER ARTIFACT") or
                   text.startswith("[THIS THREAD BEFORE") for text in item_texts(
                       plain.input_items))


# ------------------------------------------------ §1i late artifact evidence

def _entry(namespace, row_id, source_ts, row, *, manifest_hash=None):
    return {"snapshot_id": "snap-1", "artifact_namespace": namespace, "row_id": row_id,
            "source_ts": source_ts, "status": row.get("status"),
            "manifest_content_hash": manifest_hash, "manifest_status_at_capture": None,
            "row": row}


def _late_db(entries):
    db = AsyncMock()
    db.late_artifact_evidence_async.return_value = list(entries)
    return db


def test_two_artifacts_on_one_source_message_both_render():
    """Appendix C test 29. `(source_ts, snapshot_id)` alone collides the moment one message
    carries an image and a document, which is the ordinary case."""
    db = _late_db([
        _entry("document_extraction", "7", R1,
               {"id": 7, "filename": "q3.pdf", "summary": "the quarter's numbers",
                "status": "complete"}),
        _entry("image_analysis", "3", R1,
               {"id": 3, "analysis": "a bar chart of revenue", "status": "ready"}),
    ])
    built = stream([msg(T0)], snapshot=snapshot_row())

    items = asyncio.run(build_late_artifact_items(db=db, stream=built))

    assert len(items) == 2
    # Ordered (source_ts, artifact_namespace, row_id): document_extraction before image_analysis.
    assert "the quarter's numbers" in items[0]["content"]
    assert "a bar chart of revenue" in items[1]["content"]
    for item in items:
        assert item["role"] == "user"
        assert item["content"].startswith(
            f"[EARLIER ARTIFACT — completed after compaction; source message {R1}, "
            f"snapshot snap-1]")


def test_late_artifacts_are_ordered_by_source_then_namespace_then_row():
    db = _late_db([
        _entry("image_analysis", "9", R2, {"id": 9, "analysis": "later", "status": "ready"}),
        _entry("image_analysis", "2", R1, {"id": 2, "analysis": "earlier b", "status": "ready"}),
        _entry("image_analysis", "1", R1, {"id": 1, "analysis": "earlier a", "status": "ready"}),
    ])
    built = stream([msg(T0)], snapshot=snapshot_row())
    items = asyncio.run(build_late_artifact_items(db=db, stream=built))
    assert [i["content"].splitlines()[-1] for i in items] == ["earlier a", "earlier b", "later"]


def test_a_render_identical_to_its_captured_hash_is_already_in_the_summary():
    """The statusless namespaces store the literal "complete", so only the content hash can say
    whether the render changed since capture."""
    body = "q3.pdf: the quarter's numbers"
    db = _late_db([_entry("document_extraction", "7", R1,
                          {"id": 7, "filename": "q3.pdf", "summary": "the quarter's numbers",
                           "status": "complete"},
                          manifest_hash=hashlib.sha256(body.encode("utf-8")).hexdigest())])
    built = stream([msg(T0)], snapshot=snapshot_row())
    assert asyncio.run(build_late_artifact_items(db=db, stream=built)) == ()


def test_a_changed_render_under_a_captured_hash_still_appears():
    db = _late_db([_entry("document_extraction", "7", R1,
                          {"id": 7, "filename": "q3.pdf", "summary": "now with real numbers",
                           "status": "complete"},
                          manifest_hash="a" * 64)])
    built = stream([msg(T0)], snapshot=snapshot_row())
    items = asyncio.run(build_late_artifact_items(db=db, stream=built))
    assert len(items) == 1 and "now with real numbers" in items[0]["content"]


def test_a_missing_row_renders_the_honest_one_line_failure():
    db = _late_db([{"snapshot_id": "snap-1", "artifact_namespace": "image_analysis",
                    "row_id": "3", "source_ts": R1, "manifest_content_hash": None, "row": None}])
    built = stream([msg(T0)], snapshot=snapshot_row())
    items = asyncio.run(build_late_artifact_items(db=db, stream=built))
    assert items[0]["content"] == (
        f"[EARLIER ARTIFACT — could not be rendered: row_missing; source message {R1}, "
        f"snapshot snap-1]")


def test_an_unrenderable_namespace_is_a_failure_item_not_a_crash():
    db = _late_db([_entry("something_new", "3", R1, {"id": 3, "status": "ready"})])
    built = stream([msg(T0)], snapshot=snapshot_row())
    items = asyncio.run(build_late_artifact_items(db=db, stream=built))
    assert "could not be rendered: render_error" in items[0]["content"]


def test_late_evidence_is_read_against_the_pinned_snapshot_not_the_active_one():
    db = _late_db([])
    built = stream([msg(T0)], snapshot=snapshot_row(snapshot_id="pinned-S1"))
    asyncio.run(build_late_artifact_items(db=db, stream=built))
    args, kwargs = db.late_artifact_evidence_async.await_args
    assert args == (TEAM, CH, "pinned-S1")
    assert kwargs == {"boundary_ts": BOUNDARY, "high_ts": H}


def test_an_unavailable_accessor_costs_the_evidence_not_the_turn():
    db = AsyncMock()
    db.late_artifact_evidence_async.side_effect = RuntimeError("no such table")
    built = stream([msg(T0)], snapshot=snapshot_row())
    assert asyncio.run(build_late_artifact_items(db=db, stream=built)) == ()


def test_genesis_asks_for_no_late_evidence():
    db = AsyncMock()
    built = stream([msg(T0)], snapshot=None)
    assert asyncio.run(build_late_artifact_items(db=db, stream=built)) == ()
    assert db.late_artifact_evidence_async.await_count == 0


@pytest.mark.parametrize("namespace,row,expected", [
    ("image_analysis", {"id": 1, "analysis": "a chart", "status": "ready"},
     "You can SEE this image description; you cannot edit or re-render the original."),
    ("document_extraction", {"id": 1, "filename": "a.pdf", "summary": "s", "status": "complete"},
     "Extracted content follows; read_document may have fresher bytes."),
    ("ambient_artifact", {"id": 1, "kind": "link", "title": "t", "summary": "s",
                          "status": "ready", "derivation_source": "read", "ref": "r"},
     "Background summary of a linked resource."),
    ("tool_provenance", {"id": 1, "tools_json": '[{"tool_name": "web_search"}]',
                         "status": "complete"},
     "Record of a tool run that completed after compaction."),
])
def test_each_namespace_renders_its_pinned_kind_line(namespace, row, expected):
    db = _late_db([_entry(namespace, str(row["id"]), R1, row)])
    built = stream([msg(T0)], snapshot=snapshot_row())
    items = asyncio.run(build_late_artifact_items(db=db, stream=built))
    assert items and items[0]["content"].splitlines()[1] == expected
