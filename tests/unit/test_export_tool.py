"""export_conversation — collecting a whole conversation into the sandbox.

The properties under test are the ones that make an export trustworthy: the canonical
authorization gate, complete-or-nothing paging (dedupe, threads under out-of-range roots),
chunking at the transfer bound rather than truncation, and the dead-container guard.
"""
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor import export_tool
from tool_registry import ToolContext, ToolRegistry

CHANNEL = "C0EXPORT1"


@pytest.fixture(autouse=True)
def _no_page_pause(monkeypatch):
    # The 1.2s production cadence is a rate-limit courtesy, not behavior under test.
    monkeypatch.setattr(export_tool, "_PAGE_PAUSE_S", 0.0)


def _msg(ts: str, user: str = "U1", text: str = "hello", **kw: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {"ts": ts, "user": user, "text": text}
    base.update(kw)
    return base


def _page(messages: List[Dict[str, Any]], cursor: str = "") -> Dict[str, Any]:
    page: Dict[str, Any] = {"ok": True, "messages": messages}
    if cursor:
        page["response_metadata"] = {"next_cursor": cursor}
        page["has_more"] = True
    return page


class _Web:
    """A Slack web-client double that records the params each call was made with."""

    def __init__(self, history_pages, replies_pages=None):
        self._history = list(history_pages)
        self._replies = dict(replies_pages or {})
        self.history_calls: List[Dict[str, Any]] = []
        self.replies_calls: List[Dict[str, Any]] = []

    async def conversations_history(self, **kwargs):
        self.history_calls.append(kwargs)
        return self._history[min(len(self.history_calls) - 1, len(self._history) - 1)]

    async def conversations_replies(self, **kwargs):
        self.replies_calls.append(kwargs)
        return self._replies.get(kwargs.get("ts"), _page([]))


def _client(web: _Web, verdict: str = "ALLOW", names: Optional[Dict[str, str]] = None):
    client = MagicMock()
    client.app = SimpleNamespace(client=web)
    client._authorize_channel_read = AsyncMock(return_value=(verdict, "both_members"))
    client.resolve_usernames = AsyncMock(return_value=dict(names or {}))
    client.classify_sender = MagicMock(return_value="human")
    client._text_with_supplementary = MagicMock(side_effect=lambda m, c: m.get("text") or "")
    return client


def _ctx(client, container="cntr_x", gone=None):
    created: List[Any] = []

    async def _create(container_id, file):
        created.append((container_id, file.name, file.getvalue()))
        return SimpleNamespace(id=f"cfile_{len(created)}",
                               path=f"/mnt/data/{file.name}")

    raw = MagicMock()
    raw.containers.files.create = AsyncMock(side_effect=_create)
    processor = MagicMock()
    processor.openai_client.client = raw

    ctx = ToolContext(channel_id=CHANNEL, thread_ts="1.0", client=client, processor=processor,
                      container_id=container, container_gone_sink=list(gone or []),
                      mounted_files=[], user_id="U1", requester_is_human=True)
    return ctx, created


def _lines(blob: bytes) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in blob.decode("utf-8").splitlines()]


@pytest.mark.unit
@pytest.mark.asyncio
class TestAuthorization:
    async def test_a_denied_read_returns_the_canonical_refusal_and_stages_nothing(self):
        from slack_client.history_tool import ACCESS_DENIED_MESSAGE

        web = _Web([_page([_msg("100.0")])])
        client = _client(web, verdict="DENY")
        ctx, created = _ctx(client)

        result = await export_tool.execute_export_conversation(ctx, {})

        assert result["ok"] is False and result["error"] == "not_accessible"
        # Byte-identical to the history tools' refusal: a varying message is an existence oracle.
        assert result["message"] == ACCESS_DENIED_MESSAGE
        assert created == [] and web.history_calls == []

    async def test_a_redirect_is_indistinguishable_from_a_denial(self):
        web = _Web([_page([_msg("100.0")])])
        deny = await export_tool.execute_export_conversation(_ctx(_client(web, "DENY"))[0], {})
        redirect = await export_tool.execute_export_conversation(
            _ctx(_client(web, "REDIRECT"))[0], {})
        assert deny == redirect

    async def test_the_gate_is_asked_about_the_channel_the_model_named(self):
        web = _Web([_page([_msg("100.0")])])
        client = _client(web, verdict="DENY")
        ctx, _ = _ctx(client)
        await export_tool.execute_export_conversation(ctx, {"channel_id": "C0OTHER99"})
        assert client._authorize_channel_read.await_args.args[0] == "C0OTHER99"


@pytest.mark.unit
@pytest.mark.asyncio
class TestSandboxGuards:
    async def test_a_recycled_container_refuses_before_any_paging(self):
        web = _Web([_page([_msg("100.0")])])
        client = _client(web)
        ctx, created = _ctx(client, container="cntr_dead", gone=["cntr_dead"])

        result = await export_tool.execute_export_conversation(ctx, {})

        assert result["ok"] is False and result["error"] == "container_recycled"
        assert created == [] and web.history_calls == []

    async def test_no_addressable_container_is_an_honest_refusal(self):
        web = _Web([_page([_msg("100.0")])])
        ctx, created = _ctx(_client(web), container=None)

        result = await export_tool.execute_export_conversation(ctx, {})

        assert result["ok"] is False and result["error"] == "sandbox_unavailable"
        assert created == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestCollection:
    async def test_it_follows_the_cursor_and_stages_every_page(self):
        web = _Web([
            _page([_msg("300.0", text="third"), _msg("200.0", text="second")], cursor="c1"),
            _page([_msg("100.0", text="first")]),
        ])
        ctx, created = _ctx(_client(web))

        result = await export_tool.execute_export_conversation(ctx, {})

        assert result["ok"] is True and result["message_count"] == 3
        assert len(web.history_calls) == 2
        assert web.history_calls[0]["limit"] == 200
        assert web.history_calls[1]["cursor"] == "c1"
        # Oldest first: a transcript is read top to bottom.
        assert [line["text"] for line in _lines(created[0][2])] == ["first", "second", "third"]

    async def test_a_thread_root_is_not_exported_twice(self):
        root = _msg("100.0", text="root", reply_count=2, latest_reply="120.0")
        web = _Web(
            [_page([root])],
            {"100.0": _page([dict(root), _msg("110.0", text="reply", thread_ts="100.0")])},
        )
        ctx, created = _ctx(_client(web))

        result = await export_tool.execute_export_conversation(ctx, {})

        assert result["message_count"] == 2 and result["thread_count"] == 1
        assert [line["ts"] for line in _lines(created[0][2])] == ["100.0", "110.0"]

    async def test_a_bounded_export_walks_a_thread_whose_root_is_older_than_the_bound(self):
        # The case bounded history cannot see: a reply IN range under a root OUT of range.
        old_root = _msg("100.0", text="ancient", reply_count=1, latest_reply="900.0")
        web = _Web(
            [_page([_msg("800.0", text="recent"), old_root])],
            {"100.0": _page([_msg("900.0", text="fresh reply", thread_ts="100.0")])},
        )
        ctx, created = _ctx(_client(web))

        result = await export_tool.execute_export_conversation(ctx, {"oldest": "500.0"})

        # `oldest` is withheld from the history call so the older root is still visible…
        assert web.history_calls[0].get("oldest") is None
        # …and the root itself is NOT exported, only its in-range reply.
        assert [line["text"] for line in _lines(created[0][2])] == ["recent", "fresh reply"]
        assert result["message_count"] == 2

    async def test_a_thread_that_died_before_the_bound_is_not_walked(self):
        stale = _msg("100.0", reply_count=4, latest_reply="150.0")
        web = _Web([_page([_msg("800.0"), stale])], {"100.0": _page([_msg("110.0")])})
        ctx, _ = _ctx(_client(web))

        await export_tool.execute_export_conversation(ctx, {"oldest": "500.0"})

        assert web.replies_calls == []

    async def test_include_threads_false_bounds_the_history_call_itself(self):
        web = _Web([_page([_msg("800.0", reply_count=3, latest_reply="900.0")])])
        ctx, _ = _ctx(_client(web))

        await export_tool.execute_export_conversation(
            ctx, {"oldest": "500.0", "include_threads": False})

        assert web.history_calls[0]["oldest"] == "500.000000"
        assert web.replies_calls == []

    async def test_a_refused_page_stages_nothing_rather_than_a_partial_export(self):
        from slack_sdk.errors import SlackApiError

        web = _Web([_page([_msg("300.0")], cursor="c1")])

        async def _boom(**kwargs):
            web.history_calls.append(kwargs)
            if len(web.history_calls) == 1:
                return _page([_msg("300.0")], cursor="c1")
            raise SlackApiError("nope", SimpleNamespace(
                data={"error": "channel_not_found"},
                get=lambda k, d=None: {"error": "channel_not_found"}.get(k, d),
                headers={}))

        web.conversations_history = _boom  # type: ignore[method-assign]
        ctx, created = _ctx(_client(web))

        result = await export_tool.execute_export_conversation(ctx, {})

        assert result["ok"] is False and result["error"] == "history_unavailable"
        assert created == []

    async def test_an_empty_range_says_so_instead_of_staging_an_empty_file(self):
        web = _Web([_page([])])
        ctx, created = _ctx(_client(web))

        result = await export_tool.execute_export_conversation(ctx, {})

        assert result["ok"] is False and result["error"] == "empty_export"
        assert created == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestSerialization:
    async def test_a_line_carries_the_facts_the_analysis_needs(self):
        web = _Web([_page([_msg(
            "100.0", user="U7", text="ship it", thread_ts="90.0",
            reactions=[{"name": "tada", "count": 2, "users": ["U1", "U2"]}],
            files=[{"name": "notes.pdf", "mimetype": "application/pdf"}])])])
        client = _client(web, names={"U7": "Riley Reyes"})
        ctx, created = _ctx(client)

        result = await export_tool.execute_export_conversation(ctx, {})

        line = _lines(created[0][2])[0]
        assert line == {"ts": "100.0", "thread_ts": "90.0", "user": "Riley Reyes",
                        "sender": "human", "text": "ship it",
                        "reactions": [{"emoji": "tada", "count": 2, "users": ["U1", "U2"]}],
                        "files": [{"name": "notes.pdf", "mimetype": "application/pdf"}]}
        assert result["format"] == "jsonl"

    async def test_a_bot_author_keeps_its_own_name_and_sender_class(self):
        web = _Web([_page([{"ts": "100.0", "bot_id": "B9", "username": "Deploybot",
                            "text": "build green"}])])
        client = _client(web)
        client.classify_sender = MagicMock(return_value="other_bot")
        ctx, created = _ctx(client)

        await export_tool.execute_export_conversation(ctx, {})

        line = _lines(created[0][2])[0]
        assert line["user"] == "Deploybot" and line["sender"] == "other_bot"

    async def test_the_digest_is_recorded_so_the_export_cannot_be_posted_back_out(self):
        web = _Web([_page([_msg("100.0")])])
        ctx, _ = _ctx(_client(web))

        await export_tool.execute_export_conversation(ctx, {})

        from message_processor.file_mount import mounted_digests
        assert len(mounted_digests(ctx)) == 1


@pytest.mark.unit
@pytest.mark.asyncio
class TestChunking:
    async def test_an_oversized_export_is_split_into_parts_and_never_truncated(self, monkeypatch):
        from config import config as cfg

        # One message per part: the smallest bound that still exercises the split.
        monkeypatch.setattr(export_tool, "_max_transfer_bytes", lambda: 80)
        assert cfg.artifact_max_mb  # the real bound is config's; this only shrinks it
        web = _Web([_page([_msg(f"{100 + i}.0", text="x" * 40) for i in range(3)])])
        ctx, created = _ctx(_client(web))

        result = await export_tool.execute_export_conversation(ctx, {})

        assert len(created) == 3 and len(result["paths"]) == 3
        assert [name for _, name, _ in created] == [
            "export-part-001.jsonl", "export-part-002.jsonl", "export-part-003.jsonl"]
        # Every message survives the split.
        exported = [line for _, _, blob in created for line in _lines(blob)]
        assert len(exported) == 3 and result["message_count"] == 3

    async def test_a_single_line_over_the_bound_is_its_own_part(self):
        parts = export_tool._chunk([b"a" * 50 + b"\n", b"b\n"], 10)
        assert parts == [b"a" * 50 + b"\n", b"b\n"]

    async def test_a_failed_upload_is_a_result_not_an_exception(self):
        web = _Web([_page([_msg("100.0")])])
        ctx, _ = _ctx(_client(web))
        ctx.processor.openai_client.client.containers.files.create = AsyncMock(
            side_effect=RuntimeError("container gone"))

        result = await export_tool.execute_export_conversation(ctx, {})

        assert result["ok"] is False and result["error"] == "stage_failed"


@pytest.mark.unit
class TestRegistration:
    def test_it_is_hidden_when_the_sandbox_is_off(self, monkeypatch):
        registry = ToolRegistry()
        export_tool.register_export_tool(registry)
        assert "export_conversation" in {s["name"] for s in registry.schemas({})}
        assert "export_conversation" in {
            s["name"] for s in registry.schemas({}, surface="channel")}

        off = {"enable_code_interpreter": False}
        assert "export_conversation" not in {s["name"] for s in registry.schemas(off)}
        assert "export_conversation" not in {
            s["name"] for s in registry.schemas(off, surface="channel")}

    def test_the_timeout_is_the_walk_s_not_the_registry_default(self):
        registry = ToolRegistry()
        export_tool.register_export_tool(registry)
        assert registry._tools["export_conversation"]["timeout"] == 600.0
