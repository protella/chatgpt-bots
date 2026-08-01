"""A pinned channel turn, for tests that drive the handlers directly.

A channel turn's request is a function of state pinned before the handlers run: the serialized
stream, H, the steering snapshot, the capability profile, this turn's raw attachment parts. In
production `MessageProcessor.process_message` pins all of it (base.py steps 1-10). A test that
calls `_handle_text_response` or `_handle_streaming_text_response` straight has skipped that, and
the assembler refuses rather than inventing a window — which is the right production behaviour and
a nuisance in a test, so this builds the pins in one line.

`pin_channel_turn(turn, ...)` gives a turn a real ChannelStream over synthetic NormalizedMessages
— a real serializer run, real pinned records, real bytes. Nothing here fakes the assembler's
input; it only supplies what Slack and the database would have supplied.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from message_processor import channel_request, channel_stream
from message_processor.channel_stream import (InventoryPin, PinnedTuple, ReceiptRec,
                                              SidecarPin, classify_chrome,
                                              serialize_stream,
                                              serializer_config_snapshot)
from message_processor.channel_steering import ChannelSteeringSnapshot
from slack_client.normalizer import FileRef, NormalizedMessage, ORIGIN_HISTORY, ReactionRec

DEFAULT_CHANNEL = "C1"
DEFAULT_TEAM = "T1"
DEFAULT_MODEL = "gpt-5.6-sol"


def normalized(ts: str, text: str = "hello", *, sender_id: str = "U1",
               sender_type: str = "human", channel_id: str = DEFAULT_CHANNEL,
               team_id: str = DEFAULT_TEAM, thread_root_ts: Optional[str] = None,
               files: Sequence[FileRef] = (), reactions: Sequence[ReactionRec] = (),
               raw_bot_name: Optional[str] = None, edited_ts: Optional[str] = None,
               is_broadcast: bool = False, is_tombstone: bool = False,
               reply_count: Optional[int] = None, latest_reply: Optional[str] = None,
               mention_ids: Sequence[str] = ()) -> NormalizedMessage:
    """One synthetic message, in the shape the normalizer produces."""
    return NormalizedMessage(
        team_id=team_id, channel_id=channel_id, ts=ts, thread_root_ts=thread_root_ts,
        subtype=None, sender_id=sender_id, sender_type=sender_type, raw_bot_name=raw_bot_name,
        text=text, files=tuple(files), reactions=tuple(reactions), edited_ts=edited_ts,
        is_broadcast=is_broadcast, is_tombstone=is_tombstone, reply_count=reply_count,
        latest_reply=latest_reply, mention_ids=tuple(mention_ids), origin=ORIGIN_HISTORY)


def file_ref(file_id: str = "F1", name: str = "report.pdf",
             mimetype: str = "application/pdf", size: Optional[int] = 1024,
             url: str = "https://files.slack.com/files-pri/T1-F1/report.pdf",
             kind: str = "file") -> FileRef:
    return FileRef(id=file_id, name=name, mimetype=mimetype, size=size, url_private=url,
                   kind=kind)


def sidecars(*, window: Tuple[str, bool] = ("1.0", True),
             receipts: Sequence[ReceiptRec] = (),
             coverage_start: str = "1.0", coverage_status: str = "complete",
             coverage_reason: Optional[str] = "genesis",
             image_analyses: Sequence[Dict[str, Any]] = (),
             document_extractions: Sequence[Dict[str, Any]] = (),
             ambient_artifacts: Sequence[Dict[str, Any]] = (),
             tool_usage: Sequence[Tuple[str, Tuple[Dict[str, Any], ...]]] = (),
             receipt_feature_epoch_ts: Optional[str] = None) -> SidecarPin:
    return SidecarPin(
        window=window, receipts=tuple(receipts),
        receipt_feature_epoch_ts=receipt_feature_epoch_ts,
        coverage=InventoryPin(start_ts=coverage_start, status=coverage_status,
                              reason=coverage_reason),
        activity_roots=(), activity_event_ts=(),
        image_analyses=tuple(image_analyses), document_extractions=tuple(document_extractions),
        ambient_artifacts=tuple(ambient_artifacts), tool_usage=tuple(tool_usage),
        versions_hash="sidecars-hash")


def build_stream(messages: Iterable[NormalizedMessage], *, h: str = "9999.0",
                 channel_id: str = DEFAULT_CHANNEL, team_id: str = DEFAULT_TEAM,
                 actor_map: Optional[Sequence[Tuple[str, str]]] = None,
                 pinned_sidecars: Optional[SidecarPin] = None,
                 origin_root_ts: Optional[str] = None,
                 origin_messages: Optional[Sequence[NormalizedMessage]] = None,
                 reach_tools: Sequence[str] = ()):
    """A real ChannelStream over synthetic messages — the serializer actually runs.

    `origin_root_ts` + `origin_messages` build the POST-BREAKPOINT origin block. They default to
    empty, so every existing caller gets the periphery-only stream it already had; a test that
    wants the origin block asks for it explicitly rather than having one appear underneath it.
    """
    ordered = tuple(sorted(messages, key=lambda m: float(m.ts)))
    origin = tuple(sorted(origin_messages or (), key=lambda m: float(m.ts)))
    pins = pinned_sidecars if pinned_sidecars is not None else sidecars()
    if actor_map is None:
        names: Dict[str, str] = {}
        for message in (*ordered, *origin):
            if message.sender_id:
                names.setdefault(message.sender_id,
                                 message.raw_bot_name or f"user-{message.sender_id}")
        actor_map = tuple(sorted(names.items()))
    floor = pins.window[0] if pins.window[0] != "0" else ""
    # The chrome memo covers the DEDUPED UNION of both snapshots, preferring the periphery copy —
    # exactly what `__post_init__` recomputes to validate against.
    union = list(ordered) + [m for m in origin if m.ts not in {p.ts for p in ordered}]
    pinned = PinnedTuple(
        team_id=team_id, channel_id=channel_id, window=(floor, True), H=h,
        periphery_floor_ts=floor, selection_version=1,
        chrome_ts=classify_chrome(tuple(union),
                                  chrome_markers=serializer_config_snapshot()["chrome_markers"]),
        fetch_snapshot=ordered, sidecar_versions_hash=pins.versions_hash,
        actor_map=tuple(actor_map), actor_map_hash="actor-hash",
        serializer_version=channel_stream.SERIALIZER_VERSION,
        serializer_config_hash="serializer-config-hash",
        capability_profile_hash="capability-hash", tool_schema_version="tools-v1",
        coverage=pins.coverage, receipt_feature_epoch_ts=pins.receipt_feature_epoch_ts,
        receipt_map=tuple((r.ts, r.state, r.turn_id, r.thread_root_ts) for r in pins.receipts),
        sidecars=pins,
        origin_root_ts=origin_root_ts, origin_snapshot=origin,
        reach_tools=tuple(reach_tools))
    return serialize_stream(pinned)


def steering(policy: Optional[str] = None,
             facts: Optional[str] = None) -> ChannelSteeringSnapshot:
    return ChannelSteeringSnapshot(developer_policy=policy, user_facts=facts)


def thread_config(**overrides) -> Dict[str, Any]:
    cfg = {"model": DEFAULT_MODEL, "temperature": 1.0, "max_tokens": 4000,
           "reasoning_effort": "medium", "verbosity": "medium", "enable_web_search": False,
           "enable_code_interpreter": False, "enable_streaming": True,
           "custom_instructions": None}
    cfg.update(overrides)
    return cfg


def pin_channel_turn(turn, *, messages: Optional[Sequence[NormalizedMessage]] = None,
                     stream=None, trigger_ts: str = "10.0",
                     origin_thread_ts: Optional[str] = "10.0",
                     channel_id: str = DEFAULT_CHANNEL, team_id: str = DEFAULT_TEAM,
                     trigger_text: str = "hi", h: str = "9999.0",
                     steering_snapshot=None, config: Optional[Dict[str, Any]] = None,
                     image_parts: Sequence[Dict[str, Any]] = (),
                     file_parts: Sequence[Dict[str, Any]] = (),
                     document_inputs: Sequence[Dict[str, Any]] = (),
                     failed_attachment_names: Sequence[str] = (),
                     batched_image_parts: Sequence[Dict[str, Any]] = (),
                     cohort_sources: Sequence[channel_request.CohortSource] = (),
                     requester: Optional[channel_request.RequesterFacts] = None,
                     channel_info: Optional[Dict[str, Any]] = None,
                     num_members: Optional[int] = None,
                     origin_participants: Optional[Dict[str, str]] = None,
                     wake_source: Optional[str] = None,
                     queued_batch_size: Optional[int] = None,
                     origin_messages: Optional[Sequence[NormalizedMessage]] = None,
                     prepared: Any = None):
    """Pin everything a channel turn needs, exactly where base.py pins it.

    `prepared` lets a test skip the tool/catalog resolution the way base.py's own pin does; pass
    the five-tuple `(registry, request_config, no_reply_available, contract_suffix, ci_container)`
    to control the tools half.
    """
    if stream is None:
        stream = build_stream(
            messages if messages is not None else [normalized(trigger_ts, trigger_text)],
            h=h, channel_id=channel_id, team_id=team_id,
            origin_root_ts=origin_thread_ts if origin_messages else None,
            origin_messages=origin_messages)
    ctx = channel_request.ChannelTurnContext(
        stream=stream,
        steering=steering_snapshot if steering_snapshot is not None else steering(),
        thread_config=config if config is not None else thread_config(),
        channel_id=channel_id, team_id=team_id, trigger_ts=trigger_ts,
        origin_thread_ts=origin_thread_ts, trigger_text=trigger_text,
        image_parts=tuple(image_parts), file_parts=tuple(file_parts),
        document_inputs=tuple(document_inputs),
        failed_attachment_names=tuple(failed_attachment_names),
        batched_image_parts=tuple(batched_image_parts),
        cohort_sources=tuple(cohort_sources),
        canonical_files=channel_request.canonical_files_from_stream(stream),
        origin_participants=dict(origin_participants or {}),
        requester=requester if requester is not None else channel_request.RequesterFacts(
            user_id="U1", real_name="Alice", sender_type="human"),
        channel_info=channel_info, num_members=num_members,
        wake_source=wake_source, queued_batch_size=queued_batch_size)
    turn.channel_stream = stream
    turn.channel_turn_context = ctx
    turn.stream_build_present = True
    turn.H = stream.pinned.H
    if prepared is not None:
        turn.channel_prepared = prepared
    return ctx


def no_tools_prepared(request_config: Optional[Dict[str, Any]] = None,
                      contract_suffix: Optional[str] = None):
    """The `channel_prepared` tuple for a turn with no local tools — the cheap default."""
    return (None, dict(request_config or {}), False, contract_suffix, None)


def item_texts(input_items: Sequence[Dict[str, Any]]) -> List[str]:
    """Every rendered string in a request's items, so a test can assert on what the model reads."""
    out: List[str] = []
    for item in input_items:
        content = item.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_text":
                    out.append(str(part.get("text") or ""))
    return out
