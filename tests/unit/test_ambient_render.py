"""F51 — render seams: thread-history injection (batched, user-scoped), pulse composition in all
three renderers + deletion, and the fetch_url tool."""
import sqlite3
import tempfile
import types

import pytest

import ambient_fetch
from database import DatabaseManager
from message_processor.utilities import MessageUtilitiesMixin, _render_ambient_artifact
from slack_client.channel_pulse import ChannelPulse

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = DatabaseManager("test")
        d.db_path = f"{tmpdir}/test.db"
        d.conn = sqlite3.connect(d.db_path, check_same_thread=False, isolation_level=None)
        d.conn.row_factory = sqlite3.Row
        d.init_schema()
        yield d


class _Harness(MessageUtilitiesMixin):
    def __init__(self, db):
        self.db = db

    def log_debug(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass


class _CountingDB:
    """Fake db that counts ambient batch loads to prove the render path is NOT N+1."""

    def __init__(self):
        self.ambient_calls = 0
        self.image_calls = 0

    async def get_ambient_artifacts_for_messages(self, channel_id, ts_list, statuses=None):
        self.ambient_calls += 1
        return {"1.1": [{"kind": "link", "title": "T", "summary": "S1",
                         "derivation_source": "fetch"}],
                "2.2": [{"kind": "file", "title": "doc.pdf", "summary": "S2",
                         "derivation_source": "document"}]}

    async def get_images_by_message_async(self, thread_key, ts):
        self.image_calls += 1
        return []


async def test_inject_ambient_is_single_batched_query_user_scoped():
    fake = _CountingDB()
    h = _Harness(fake)
    ts_state = types.SimpleNamespace(channel_id="C1", thread_ts="1.1")
    messages = [
        {"role": "user", "content": "hi", "metadata": {"ts": "1.1"}},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more", "metadata": {"ts": "2.2"}},
    ]
    out = await h._inject_image_analyses(messages, ts_state)
    # ONE ambient batch load for the whole page (never N+1).
    assert fake.ambient_calls == 1
    injected = [m for m in out if m["role"] == "user" and "Ambient context" in str(m.get("content"))]
    assert len(injected) == 2  # link + file, both user-scoped (never developer)
    assert all(m["role"] == "user" for m in injected)


class _ImageRoleDB:
    """Returns one AMBIENT and one ADDRESSED image analysis for the same message."""

    async def get_ambient_artifacts_for_messages(self, channel_id, ts_list, statuses=None):
        return {}

    async def get_images_by_message_async(self, thread_key, ts):
        return [
            {"analysis": "AMBIENT-HOSTILE-DESC", "url": "u1", "image_type": "uploaded",
             "metadata": {"ambient": True}},
            {"analysis": "ADDRESSED-DESC", "url": "u2", "image_type": "uploaded",
             "metadata": {}},
        ]


async def test_image_analyses_are_user_scoped_ambient_and_addressed():
    # An image analysis is a model-written description of user-controlled image bytes, so BOTH the
    # ambient and the addressed variant must ride as untrusted USER context — never a developer
    # instruction. Ambient carries extra "untrusted" framing (the bot never answered that image).
    h = _Harness(_ImageRoleDB())
    ts_state = types.SimpleNamespace(channel_id="C1", thread_ts="1.1")
    out = await h._inject_image_analyses(
        [{"role": "user", "content": "hi", "metadata": {"ts": "1.1"}}], ts_state)
    ambient = [m for m in out if "AMBIENT-HOSTILE-DESC" in str(m.get("content"))]
    addressed = [m for m in out if "ADDRESSED-DESC" in str(m.get("content"))]
    assert ambient and ambient[0]["role"] == "user"
    assert "untrusted" in ambient[0]["content"]
    assert addressed and addressed[0]["role"] == "user"


async def test_unfurl_link_artifact_not_double_described(db):
    # SHOULD-FIX: an unfurl-sourced link artifact repeats F48's already-rendered preview, so the
    # injection must skip it.
    await db.insert_pending_ambient_artifact(
        channel_id="C1", source_ts="1.1", conversation_ts="1.1", kind="link", ref="u")
    await db.set_ambient_artifact_ready(
        channel_id="C1", source_ts="1.1", kind="link", ref="u", title="T",
        summary="PREVIEW", model=None, derivation_source="unfurl")
    h = _Harness(db)
    ts_state = types.SimpleNamespace(channel_id="C1", thread_ts="1.1")
    out = await h._inject_image_analyses(
        [{"role": "user", "content": "hi", "metadata": {"ts": "1.1"}}], ts_state)
    assert not [m for m in out if "PREVIEW" in str(m.get("content"))]


async def test_delete_by_source_cascades_to_ambient_image_ledger(db):
    # MUST-FIX 4: the ambient vision worker dual-writes the analysis into `images`; deleting the
    # source message must purge BOTH tables or the description lingers and keeps being injected.
    await db.insert_pending_ambient_artifact(
        channel_id="C1", source_ts="100.0", conversation_ts="100.0", kind="image", ref="F1")
    await db.set_ambient_artifact_ready(
        channel_id="C1", source_ts="100.0", kind="image", ref="F1", title="k.png",
        summary="desc", model="m", derivation_source="vision_worker")
    await db.save_image_metadata_async(
        thread_id="C1:100.0", url="https://files/f1", image_type="uploaded",
        analysis="desc", metadata={"ambient": True, "file_id": "F1", "channel_id": "C1"},
        message_ts="100.0")
    # sanity: the dual-write is present
    assert await db.get_images_by_message_async("C1:100.0", "100.0")
    await db.delete_ambient_artifacts_by_source("C1", "100.0")
    assert not await db.get_ambient_artifacts_for_messages("C1", ["100.0"])
    assert not await db.get_images_by_message_async("C1:100.0", "100.0")


async def test_cascade_exact_match_spares_lookalikes(db):
    # MUST-FIX 2: the cascade must be EXACT — {"ambient":false} and {"description":"ambient"} and
    # a different channel_id must all survive a delete for (C1, 100.0).
    await db.save_image_metadata_async(
        thread_id="C1:100.0", url="u_true", image_type="uploaded", analysis="a",
        metadata={"ambient": True, "file_id": "F1", "channel_id": "C1"}, message_ts="100.0")
    await db.save_image_metadata_async(
        thread_id="C1:100.0", url="u_false", image_type="uploaded", analysis="a",
        metadata={"ambient": False}, message_ts="100.0")
    await db.save_image_metadata_async(
        thread_id="C1:100.0", url="u_desc", image_type="uploaded", analysis="a",
        metadata={"description": "ambient"}, message_ts="100.0")
    await db.save_image_metadata_async(
        thread_id="C2:100.0", url="u_other_chan", image_type="uploaded", analysis="a",
        metadata={"ambient": True, "file_id": "F1", "channel_id": "C2"}, message_ts="100.0")
    await db.delete_ambient_artifacts_by_source("C1", "100.0")
    surviving = {r["url"] for r in await db.get_images_by_message_async("C1:100.0", "100.0")}
    assert surviving == {"u_false", "u_desc"}  # only the true C1-ambient row was deleted
    assert {r["url"] for r in await db.get_images_by_message_async("C2:100.0", "100.0")} == {"u_other_chan"}


async def test_file_id_cascade_no_substring_cross_delete(db):
    # MUST-FIX 2: file-id deletion must match the STORED id exactly — an id that is a substring
    # of another row's id/url must not be cross-deleted.
    await db.save_image_metadata_async(
        thread_id="C1:1.0", url="https://files/aaaF1bbb", image_type="uploaded", analysis="a",
        metadata={"ambient": True, "file_id": "F1AB", "channel_id": "C1"}, message_ts="1.0")
    await db.save_image_metadata_async(
        thread_id="C1:2.0", url="https://files/x", image_type="uploaded", analysis="a",
        metadata={"ambient": True, "file_id": "F1", "channel_id": "C1"}, message_ts="2.0")
    await db.delete_ambient_artifacts_by_file_id("F1")
    # Only the exact-id row (message 2.0) is gone; the F1AB row (whose url even contains "F1") stays.
    assert await db.get_images_by_message_async("C1:1.0", "1.0")
    assert not await db.get_images_by_message_async("C1:2.0", "2.0")


def test_render_ambient_artifact_frames_untrusted():
    link = _render_ambient_artifact({"kind": "link", "title": "T", "summary": "S",
                                     "derivation_source": "fetch"})
    assert "untrusted" in link and "not instructions" in link
    unfurl = _render_ambient_artifact({"kind": "link", "title": "", "summary": "S",
                                       "derivation_source": "unfurl"})
    assert "link preview" in unfurl


# ------------------------------------------------------------------- pulse composition

def _record(p, channel="C1", ts="1.0", thread_ts=None, text="hello"):
    p.record(channel, ts=ts, thread_ts=thread_ts, user_id="U1", display_name="Alice",
             sender_type="human", text=text, is_bot=False)


def test_pulse_upsert_appears_in_all_three_renderers():
    p = ChannelPulse(size=30)
    # A threaded reply (root 1.0) so both the channel buffer and the thread-tail ring hold it.
    _record(p, ts="1.0", text="root msg")
    _record(p, ts="2.0", thread_ts="1.0", text="a chart was posted")
    note = "[image (analyzed): benchmark chart]"
    assert p.upsert_artifacts("C1", "2.0", [note]) is True

    env = p.render_envelope("C1")
    tail = p.render_thread_tail("C1", "1.0", before_ts="9.0")
    addr = p.render_channel_addressee_tail("C1", before_ts="9.0")
    assert note in env
    assert note in tail
    assert note in addr


def test_pulse_upsert_idempotent_and_late():
    p = ChannelPulse(size=30)
    _record(p, ts="5.0", text="msg")
    note = "[link content: X]"
    assert p.upsert_artifacts("C1", "5.0", [note]) is True
    assert p.upsert_artifacts("C1", "5.0", [note]) is False  # dedup, no double-render
    assert p.render_envelope("C1").count(note) == 1


def test_pulse_remove_message_drops_entry():
    p = ChannelPulse(size=30)
    _record(p, ts="1.0", text="keep")
    _record(p, ts="2.0", text="delete me")
    assert p.remove_message("C1", "2.0") is True
    assert "delete me" not in p.render_envelope("C1")
    assert "keep" in p.render_envelope("C1")


# ------------------------------------------------------------------- fetch_url tool

def test_fetch_url_registration():
    from message_processor.fetch_url_tool import register_fetch_url_tool
    from tool_registry import ToolRegistry
    reg = ToolRegistry()
    register_fetch_url_tool(reg)
    schema = reg.get_schema("fetch_url") if hasattr(reg, "get_schema") else None
    # Fall back to a name probe if the registry exposes a different accessor.
    names = getattr(reg, "names", lambda: [])() if hasattr(reg, "names") else []
    assert schema is not None or "fetch_url" in names or "fetch_url" in getattr(reg, "_tools", {})


async def test_fetch_url_tool_returns_untrusted_and_persists(db, monkeypatch):
    from message_processor.fetch_url_tool import execute_fetch_url
    from tool_registry import ToolContext

    async def fake_fetch(url, **kw):
        return ambient_fetch.FetchResult(kind="text", final_url=url, title="Reuters",
                                         text="body " * 5000, content_type="text/html")
    monkeypatch.setattr(ambient_fetch, "fetch_url", fake_fetch)

    ctx = ToolContext(channel_id="C1", thread_ts="1.1", trigger_ts="1.1", db=db)
    out = await execute_fetch_url(ctx, {"url": "https://reuters.com/article"})
    assert out["ok"] and out["kind"] == "text"
    assert "untrusted_external_content" in out and out["has_more"] is True
    # Persisted as a link artifact keyed to the triggering message.
    rows = await db.get_ambient_artifacts_for_messages("C1", ["1.1"], statuses=["ready"])
    assert rows["1.1"][0]["kind"] == "link"


async def test_fetch_url_tool_error(db, monkeypatch):
    from message_processor.fetch_url_tool import execute_fetch_url
    from tool_registry import ToolContext

    async def fake_fetch(url, **kw):
        return ambient_fetch.FetchResult(kind="error", error_code=ambient_fetch.ERR_BLOCKED_SSRF)
    monkeypatch.setattr(ambient_fetch, "fetch_url", fake_fetch)
    ctx = ToolContext(channel_id="C1", thread_ts="1.1", trigger_ts="1.1", db=db)
    out = await execute_fetch_url(ctx, {"url": "https://10.0.0.1/x"})
    assert out["ok"] is False and out["error"] == ambient_fetch.ERR_BLOCKED_SSRF
