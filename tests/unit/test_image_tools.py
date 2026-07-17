"""F34 — image generation/editing as tools (message_processor/image_tools.py).

The properties worth defending, in rough order of how much they'd hurt to get wrong:

1. **A syntactically valid image id is not authorization.** edit_image resolves ids against
   THIS TURN's catalog snapshot; an invented id, or one belonging to another thread, must be
   an error the model can recover from — never a silent "closest match". Editing the wrong
   image is an expensive, irreversible side effect that lands in someone's Slack thread.
2. **The image model is the user's.** It appears in no schema, so the model cannot express a
   different one; an override that names one anyway is dropped and reported.
3. **The three tools have three different execution contracts**, and confusing them is how you
   get a turn that posts nothing: generate_image DETACHES (posts itself later),
   create_image_asset BLOCKS and posts NOTHING (the bytes go into the sandbox), edit_image
   BLOCKS and posts.
4. **create_image_asset is only offered when there is an addressable container.** Under
   `{"type": "auto"}` the container id is unknown until after the call, so bytes pushed
   anywhere would be invisible to the model — offering the tool guarantees a failed call.
5. **No executor may raise into the tool loop.** A moderation block is a result, not an
   exception.
"""
import base64
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from config import config
from message_processor import image_delivery, image_tools as it
from openai_client.container_errors import AUTO_CONTAINER
from openai_client.utilities import ImageData
from thread_manager import AsyncThreadStateManager
from tool_registry import ToolContext, ToolRegistry


# --------------------------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def _isolate_semaphore():
    """The process-wide image semaphore is a module global sized from config on first use."""
    it._reset_semaphore_for_tests()
    yield
    it._reset_semaphore_for_tests()


@pytest.fixture(autouse=True)
def _stub_checklist(monkeypatch):
    """Both synchronous tools post a progress checklist. It is imported inside the executor,
    so patching the module attribute swaps it everywhere; its own behavior is covered in
    test_background_image_gen.py."""
    class _Checklist:
        def __init__(self, *a, **k):
            self.steps = []

        async def step(self, text, done_text=None):
            self.steps.append(text)

    monkeypatch.setattr("message_processor.progress.ProgressChecklist", _Checklist)


@pytest.fixture(autouse=True)
def _tools_on(monkeypatch):
    # Image tools are unconditional; only the sandbox is still gated.
    monkeypatch.setattr(config, "enable_code_interpreter", True)


# ----------------------------------------------------------------------------------- helpers

def _img_bytes(fmt: str, mode: str = "RGB", color="red") -> bytes:
    buf = BytesIO()
    Image.new(mode, (4, 4), color).save(buf, format=fmt)
    return buf.getvalue()


def _png_bytes() -> bytes:
    return _img_bytes("PNG")


def _jpeg_bytes() -> bytes:
    return _img_bytes("JPEG")


def _webp_bytes() -> bytes:
    return _img_bytes("WEBP")


def _bmp_bytes() -> bytes:
    return _img_bytes("BMP")


def _animated_gif_bytes() -> bytes:
    buf = BytesIO()
    frames = [Image.new("RGB", (4, 4), "red").convert("P"),
              Image.new("RGB", (4, 4), "blue").convert("P")]
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100)
    return buf.getvalue()


# A real, Pillow-decodable PNG so the edit path's byte validation/transcode accepts it.
_SOURCE_PNG = _png_bytes()


CATALOG = [
    {"image_id": "img_7", "url": "https://files.slack.com/red-cat.png", "kind": "generated",
     "prompt": "an enhanced prompt about a red cat", "analysis": "A red cat on a blue sofa"},
    {"image_id": "img_3", "url": "https://files.slack.com/chart.png", "kind": "uploaded",
     "prompt": "", "analysis": "A bar chart of quarterly revenue"},
]


def _cfg(**over):
    base = {
        "image_model": "gpt-image-2",
        "image_size": "1024x1024",
        "image_quality": "auto",
        "image_background": "auto",
        "image_format": "png",
        "image_compression": 100,
        "input_fidelity": "high",
    }
    base.update(over)
    return base


def _image_data(fmt="png"):
    return ImageData(base64_data=base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24).decode(),
                     format=fmt, prompt="an enhanced prompt")


def _openai(*, generate=None, edit=None, create_path="/mnt/data/cover.png",
            create_error=None):
    """Stand-in OpenAI client: the two image calls plus the raw SDK's container-file surface."""
    raw = MagicMock()
    raw.containers.files.create = (
        AsyncMock(side_effect=create_error) if create_error
        else AsyncMock(return_value=SimpleNamespace(path=create_path)))
    return SimpleNamespace(
        client=raw,
        generate_image=generate or AsyncMock(return_value=_image_data()),
        edit_image=edit or AsyncMock(return_value=_image_data()),
    )


class _FakeProcessor:
    def __init__(self, openai_client=None, tm=None, schedule_error=None):
        self.openai_client = openai_client or _openai()
        self.thread_manager = tm or AsyncThreadStateManager(db=None)
        self.scheduled = []
        self.aborted = 0
        self._schedule_error = schedule_error
        self._finish_image_generation_background = AsyncMock()

    def _schedule_async_call(self, coro):
        # Record the detached job without running it (the job itself has its own coverage in
        # test_background_image_gen.py); close the coroutine so it isn't left un-awaited.
        self.scheduled.append(coro)
        coro.close()
        if self._schedule_error:
            raise self._schedule_error
        return SimpleNamespace(done=lambda: False, cancel=lambda: None)

    async def _abort_checklist(self, *a, **k):
        self.aborted += 1


class _FakeClient:
    """Slack. The point of most of these assertions is what it is NOT asked to do."""

    def __init__(self, download=_SOURCE_PNG):
        self._download = download
        self.send_image = AsyncMock()
        self.send_message = AsyncMock()
        self.download_file = AsyncMock(return_value=download)


def _ctx(processor, client=None, *, thread_config=None, container_id=None, catalog=None):
    return ToolContext(
        channel_id="C1", thread_ts="100.0", trigger_ts="100.5",
        client=client if client is not None else _FakeClient(),
        processor=processor, db=SimpleNamespace(),
        thread_config=thread_config if thread_config is not None else _cfg(),
        container_id=container_id, image_catalog=catalog,
        sandbox_image_assets=[])


def _prop_names(node):
    """Every property key anywhere in a JSON schema."""
    found = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                found |= set(value)
            found |= _prop_names(value)
    elif isinstance(node, list):
        for item in node:
            found |= _prop_names(item)
    return found


# ====================================================================== schemas (factories)

def test_v2_schema_offers_no_transparent_and_no_input_fidelity():
    schema = it.get_generate_image_schema(_cfg(image_model="gpt-image-2"))
    overrides = schema["parameters"]["properties"]["overrides"]["properties"]

    assert overrides["background"]["enum"] == ["auto", "opaque"]
    # gpt-image-2 auto-handles fidelity — advertising it would invite a param the API rejects.
    assert "input_fidelity" not in overrides
    # Free-form WxH rather than an enum, with the divisible-by-16 rule stated up front.
    assert "enum" not in overrides["size"]
    assert "DIVISIBLE BY 16" in overrides["size"]["description"]


def test_v1_schema_offers_transparent_and_input_fidelity():
    schema = it.get_generate_image_schema(_cfg(image_model="gpt-image-1"))
    overrides = schema["parameters"]["properties"]["overrides"]["properties"]

    assert "transparent" in overrides["background"]["enum"]
    assert overrides["input_fidelity"]["enum"] == ["low", "high"]
    # v1 takes only the named sizes, so they are a closed enum.
    assert overrides["size"]["enum"] == ["1024x1024", "1024x1536", "1536x1024", "auto"]


@pytest.mark.parametrize("model", ["gpt-image-1", "gpt-image-2"])
def test_no_schema_ever_lets_the_model_pick_the_image_model(model):
    cfg = _cfg(image_model=model, **{it.CATALOG_KEY: CATALOG, it.CI_CONTAINER_KEY: "cntr_x"})
    schemas = [it.get_generate_image_schema(cfg), it.get_create_image_asset_schema(cfg),
               it.get_edit_image_schema(cfg)]
    for schema in schemas:
        assert not ({"model", "image_model"} & _prop_names(schema)), schema["name"]


def test_schema_description_names_the_users_saved_defaults():
    # The model can only judge that a task warrants departing from the defaults if it is told
    # what they are.
    schema = it.get_generate_image_schema(_cfg(image_size="1024x1536", image_quality="high"))
    assert "size=1024x1536" in schema["description"]
    assert "quality=high" in schema["description"]


def test_edit_schema_is_hidden_without_a_catalog():
    # Nothing to edit → no tool. (An empty enum would be an unusable schema.)
    assert it.get_edit_image_schema(_cfg()) is None
    assert it.get_edit_image_schema(_cfg(**{it.CATALOG_KEY: []})) is None


def test_edit_schema_pins_the_ids_to_a_literal_enum():
    schema = it.get_edit_image_schema(_cfg(**{it.CATALOG_KEY: CATALOG}))
    ids = schema["parameters"]["properties"]["source_image_ids"]["items"]["enum"]

    # The ids the model may name are exactly this turn's catalog — it cannot emit another.
    assert ids == ["img_7", "img_3"]
    # …and the description says what each id IS, so the choice is informed.
    assert "A red cat on a blue sofa" in schema["description"]
    assert "(most recent)" in schema["description"]


# ====================================================================== registry gating

def _registry_names(cfg):
    registry = ToolRegistry()
    it.register_image_tools(registry)
    return {s["name"] for s in registry.schemas(cfg)}


def test_generate_image_is_always_offered():
    assert "generate_image" in _registry_names(_cfg())


def test_create_image_asset_needs_an_addressable_container():
    # No sandbox at all → hidden.
    assert "create_image_asset" not in _registry_names(_cfg())
    # An EPHEMERAL sandbox ({"type": "auto"}) → still hidden: its id is unknown until after
    # the call, so bytes pushed into it would be invisible to the code the model runs.
    assert "create_image_asset" not in _registry_names(
        _cfg(**{it.CI_CONTAINER_KEY: AUTO_CONTAINER}))
    # A real persistent container → offered.
    assert "create_image_asset" in _registry_names(_cfg(**{it.CI_CONTAINER_KEY: "cntr_abc123"}))


def test_create_image_asset_hidden_when_code_interpreter_is_off(monkeypatch):
    monkeypatch.setattr(config, "enable_code_interpreter", False)
    assert "create_image_asset" not in _registry_names(
        _cfg(**{it.CI_CONTAINER_KEY: "cntr_abc123"}))


def test_edit_image_appears_only_with_a_catalog():
    assert "edit_image" not in _registry_names(_cfg())
    assert "edit_image" in _registry_names(_cfg(**{it.CATALOG_KEY: CATALOG}))


# ====================================================================== edit_image

@pytest.mark.asyncio
@pytest.mark.critical
@pytest.mark.parametrize("bad_id", ["img_999", "img_3x", "IMG_7", "../img_7"])
async def test_edit_with_an_unresolvable_id_touches_nothing(bad_id):
    # SECURITY-RELEVANT: a syntactically valid id is not authorization. Only ids we put in
    # front of the model this turn resolve — no "did you mean the most recent one" fallback,
    # because editing the wrong image posts a wrong image into someone's thread.
    oc = _openai()
    proc = _FakeProcessor(openai_client=oc)
    client = _FakeClient()

    res = await it.execute_edit_image(
        _ctx(proc, client, catalog=CATALOG),
        {"source_image_ids": [bad_id], "prompt": "make it blue"})

    assert res == {"ok": False, "error": "unknown_image_id",
                   "message": f"No image {bad_id!r} in this thread.",
                   "valid_image_ids": ["img_7", "img_3"]}
    oc.edit_image.assert_not_awaited()      # no spend
    client.download_file.assert_not_awaited()   # not even a source fetch
    client.send_image.assert_not_awaited()      # nothing posted


@pytest.mark.asyncio
async def test_edit_fails_closed_when_one_of_several_ids_is_unknown():
    # A good id next to a bad one must not become a partial edit.
    oc = _openai()
    proc = _FakeProcessor(openai_client=oc)

    res = await it.execute_edit_image(
        _ctx(proc, catalog=CATALOG),
        {"source_image_ids": ["img_7", "img_999"], "prompt": "combine them"})

    assert res["error"] == "unknown_image_id"
    oc.edit_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_with_an_empty_catalog_is_an_error_not_a_guess():
    # The schema hides the tool with no catalog, but the executor is still dispatchable —
    # the id check is what actually protects the thread, so it must hold on its own.
    oc = _openai()
    res = await it.execute_edit_image(
        _ctx(_FakeProcessor(openai_client=oc), catalog=[]),
        {"source_image_ids": ["img_7"], "prompt": "make it blue"})

    assert res["error"] == "unknown_image_id" and res["valid_image_ids"] == []
    oc.edit_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_happy_path_posts_the_result(monkeypatch):
    publish = AsyncMock(return_value="https://files.slack.com/edited.png")
    monkeypatch.setattr(image_delivery, "publish_image", publish)
    oc = _openai()
    proc = _FakeProcessor(openai_client=oc)
    ctx = _ctx(proc, thread_config=_cfg(image_model="gpt-image-1", image_quality="high"),
               catalog=CATALOG)

    res = await it.execute_edit_image(
        ctx, {"source_image_ids": ["img_3"], "prompt": "make the bars green"})

    assert res["ok"] is True and res["status"] == "posted" and res["sources"] == ["img_3"]
    kwargs = oc.edit_image.await_args.kwargs
    assert kwargs["model"] == "gpt-image-1"          # the user's model, never the model's
    assert kwargs["quality"] == "high"
    assert kwargs["input_fidelity"] == "high"
    # The stored analysis rides along, so the edit-prompt enhancer needs no vision round-trip.
    assert kwargs["image_description"] == "A bar chart of quarterly revenue"
    # The source came from Slack, in memory, as base64 — a PNG, so it rides through unchanged
    # and is labeled with its ACTUAL mimetype (never a blanket .png guess).
    assert kwargs["input_images"] == [base64.b64encode(_SOURCE_PNG).decode()]
    assert kwargs["input_mimetypes"] == ["image/png"]
    publish.assert_awaited_once()
    # A fresh image landed in the thread → the next turn must rebuild from Slack.
    assert proc.thread_manager.consume_needs_refresh("C1:100.0") is True


@pytest.mark.asyncio
async def test_edit_reports_overrides_it_could_not_honor(monkeypatch):
    monkeypatch.setattr(image_delivery, "publish_image",
                        AsyncMock(return_value="https://files.slack.com/edited.png"))
    oc = _openai()
    res = await it.execute_edit_image(
        _ctx(_FakeProcessor(openai_client=oc), catalog=CATALOG),
        {"source_image_ids": ["img_7"], "prompt": "cut it out",
         "overrides": {"background": "transparent"}})

    # gpt-image-2 has no transparent background: the user's default is used and the model is
    # TOLD, rather than being left to believe it received a cutout.
    assert res["ok"] is True
    assert any("transparent" in note for note in res["ignored_overrides"])
    assert oc.edit_image.await_args.kwargs["background"] == "auto"


@pytest.mark.asyncio
async def test_edit_moderation_block_is_a_result_not_an_exception():
    oc = _openai(edit=AsyncMock(side_effect=Exception(
        "Your request was rejected as a result of our safety system")))
    proc = _FakeProcessor(openai_client=oc)

    res = await it.execute_edit_image(
        _ctx(proc, catalog=CATALOG),
        {"source_image_ids": ["img_7"], "prompt": "make it look like a real brand"})

    assert res["ok"] is False and res["error"] == "moderation_blocked"
    assert "safety" in res["message"].lower()
    assert proc.aborted == 1        # the progress checklist is torn down, not left spinning


@pytest.mark.asyncio
async def test_edit_reports_a_source_it_cannot_fetch():
    oc = _openai()
    client = _FakeClient()
    client.download_file = AsyncMock(return_value=None)   # Slack 404 / expired URL

    res = await it.execute_edit_image(
        _ctx(_FakeProcessor(openai_client=oc), client, catalog=CATALOG),
        {"source_image_ids": ["img_7"], "prompt": "make it blue"})

    assert res["error"] == "source_unavailable"
    oc.edit_image.assert_not_awaited()


# --- edit-source byte validation / transcoding (Finding 4) --------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected_mime", [
    (_png_bytes(), "image/png"),
    (_jpeg_bytes(), "image/jpeg"),
    (_webp_bytes(), "image/webp"),
])
async def test_edit_passes_supported_sources_through_with_their_real_mimetype(
        monkeypatch, raw, expected_mime):
    # The edit endpoint accepts png/jpeg/webp: these ride through unchanged and are labeled with
    # their ACTUAL mimetype, so edit_image names the upload part correctly instead of guessing .png.
    monkeypatch.setattr(image_delivery, "publish_image",
                        AsyncMock(return_value="https://files.slack.com/edited.png"))
    oc = _openai()
    client = _FakeClient(download=raw)

    res = await it.execute_edit_image(
        _ctx(_FakeProcessor(openai_client=oc), client, catalog=CATALOG),
        {"source_image_ids": ["img_7"], "prompt": "make it blue"})

    assert res["ok"] is True
    kwargs = oc.edit_image.await_args.kwargs
    assert kwargs["input_images"] == [base64.b64encode(raw).decode()]   # byte-identical
    assert kwargs["input_mimetypes"] == [expected_mime]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [_bmp_bytes(), _animated_gif_bytes()])
async def test_edit_transcodes_unsupported_sources_to_png(monkeypatch, raw):
    # BMP is unsupported by the edit endpoint; GIF is supported by VISION but NOT by the edit
    # endpoint. Both are transcoded to PNG in memory (first frame for the animated gif) and
    # relabeled image/png — never handed to the Images API as unsupported bytes wearing a .png name.
    monkeypatch.setattr(image_delivery, "publish_image",
                        AsyncMock(return_value="https://files.slack.com/edited.png"))
    oc = _openai()
    client = _FakeClient(download=raw)

    res = await it.execute_edit_image(
        _ctx(_FakeProcessor(openai_client=oc), client, catalog=CATALOG),
        {"source_image_ids": ["img_7"], "prompt": "make it blue"})

    assert res["ok"] is True
    kwargs = oc.edit_image.await_args.kwargs
    assert kwargs["input_mimetypes"] == ["image/png"]
    sent = base64.b64decode(kwargs["input_images"][0])
    assert sent != raw                                   # a real re-encode, not passthrough
    result = Image.open(BytesIO(sent))
    result.load()
    assert result.format == "PNG"
    assert getattr(result, "n_frames", 1) == 1           # first frame only for the animated gif


@pytest.mark.asyncio
async def test_edit_reports_an_undecodable_source_gracefully():
    # Fetched fine, but the bytes are not a decodable image. This is a graceful tool result with
    # an honest message — never unsupported bytes handed to the Images API for a raw 400.
    oc = _openai()
    client = _FakeClient(download=b"this is not an image")

    res = await it.execute_edit_image(
        _ctx(_FakeProcessor(openai_client=oc), client, catalog=CATALOG),
        {"source_image_ids": ["img_7"], "prompt": "make it blue"})

    assert res["ok"] is False and res["error"] == "unreadable_source"
    assert "img_7" in res["message"]
    oc.edit_image.assert_not_awaited()      # no spend on a source we can't send


@pytest.mark.asyncio
async def test_edit_rejects_a_source_that_balloons_past_the_byte_ceiling(monkeypatch):
    # New defect: transcoding has no output-byte ceiling — a small compressed source can expand
    # into a huge PNG and exhaust memory / exceed the API request limit. With the ceiling in place,
    # a converted source over the cap fails the same graceful three-state way as an undecodable one
    # (distinct reason), and NO API call is made. Shrink the cap so any real source trips it.
    monkeypatch.setattr(it, "_EDIT_SOURCE_MAX_BYTES", 8)
    oc = _openai()
    client = _FakeClient(download=_SOURCE_PNG)   # a valid PNG, comfortably over 8 bytes

    res = await it.execute_edit_image(
        _ctx(_FakeProcessor(openai_client=oc), client, catalog=CATALOG),
        {"source_image_ids": ["img_7"], "prompt": "make it blue"})

    assert res["ok"] is False and res["error"] == "unreadable_source"
    assert "img_7" in res["message"]
    assert "too large" in res["message"].lower()   # the distinct too_large_after_conversion reason
    oc.edit_image.assert_not_awaited()             # never handed oversized bytes to the API


@pytest.mark.asyncio
async def test_edit_requires_ids_and_a_prompt():
    proc = _FakeProcessor()
    assert (await it.execute_edit_image(
        _ctx(proc, catalog=CATALOG), {"prompt": "x"}))["error"] == "bad_arguments"
    assert (await it.execute_edit_image(
        _ctx(proc, catalog=CATALOG),
        {"source_image_ids": ["img_7"], "prompt": "  "}))["error"] == "bad_arguments"


# ====================================================================== generate_image (detached)

@pytest.mark.asyncio
@pytest.mark.critical
async def test_generate_detaches_the_job_and_claims_the_upload_latch():
    proc = _FakeProcessor()
    ctx = _ctx(proc)

    res = await it.execute_generate_image(ctx, {"prompt": "a red cat on a blue sofa"})

    assert res["ok"] is True and res["status"] == "generating"
    assert res["settings"]["model"] == "gpt-image-2"

    # Registered in flight, so a second call this turn sees it against the cap.
    in_flight = proc.thread_manager.generations_in_flight("C1:100.0")
    assert len(in_flight) == 1
    gid = in_flight[0]["generation_id"]

    # The background job was scheduled with the turn's effective settings — and NOT awaited
    # here (the whole point: the thread lock releases and the image posts itself).
    proc._finish_image_generation_background.assert_called_once()
    kwargs = proc._finish_image_generation_background.call_args.kwargs
    assert kwargs["thread_key"] == "C1:100.0" and kwargs["generation_id"] == gid
    assert kwargs["prompt"] == "a red cat on a blue sofa"
    assert kwargs["thread_config"]["image_model"] == "gpt-image-2"
    assert len(proc.scheduled) == 1

    # The upload latch is claimed NOW, under the turn's lock, so a fast follow-up "edit it"
    # cannot win the lock and target a stale image (the F1 TOCTOU fix).
    assert proc.thread_manager._upload_pending["C1:100.0"] == {gid}
    # The finalizer drops the model's ack text — the posted image IS the acknowledgment.
    assert ctx.image_generation_started is True


@pytest.mark.asyncio
async def test_generate_at_the_per_thread_cap_schedules_nothing(monkeypatch):
    monkeypatch.setattr(config, "max_concurrent_image_generations", 2)
    tm = AsyncThreadStateManager(db=None)
    tm.register_generation("C1:100.0", "gen0", "a cat")
    tm.register_generation("C1:100.0", "gen1", "a dog")
    proc = _FakeProcessor(tm=tm)
    ctx = _ctx(proc)

    res = await it.execute_generate_image(ctx, {"prompt": "a third one"})

    assert res["ok"] is False and res["error"] == "at_capacity"
    assert "2" in res["message"]                     # count-aware, so the model can say why
    assert proc.scheduled == []                      # nothing detached
    proc._finish_image_generation_background.assert_not_called()
    assert len(tm.generations_in_flight("C1:100.0")) == 2   # unchanged
    assert ctx.image_generation_started is False


@pytest.mark.asyncio
async def test_generate_passes_overrides_through_and_reports_the_rejected_ones():
    proc = _FakeProcessor()
    res = await it.execute_generate_image(_ctx(proc), {
        "prompt": "a title slide",
        "overrides": {"size": "1920x1080", "background": "transparent",
                      "model": "gpt-image-1"},
    })

    cfg = proc._finish_image_generation_background.call_args.kwargs["thread_config"]
    assert cfg["image_size"] == "1920x1088"           # snapped onto the 16px grid
    assert cfg["image_background"] == "auto"          # transparent dropped (gpt-image-2)
    assert cfg["image_model"] == "gpt-image-2"        # the user's model, not the model's pick
    notes = " ".join(res["ignored_overrides"])
    assert "1920x1088" in notes and "transparent" in notes and "fixed by the user's" in notes


@pytest.mark.asyncio
async def test_generate_unregisters_when_scheduling_fails():
    # A generation left in the registry after a failed schedule would block the thread's cap
    # forever (until the watchdog) and make the bot claim an image was coming that never is.
    proc = _FakeProcessor(schedule_error=RuntimeError("loop is closed"))
    ctx = _ctx(proc)

    res = await it.execute_generate_image(ctx, {"prompt": "a red cat"})

    assert res["ok"] is False and res["error"] == "schedule_failed"
    assert proc.thread_manager.generations_in_flight("C1:100.0") == []
    assert proc.aborted == 1
    assert ctx.image_generation_started is False


@pytest.mark.asyncio
async def test_generate_requires_a_prompt():
    proc = _FakeProcessor()
    res = await it.execute_generate_image(_ctx(proc), {"prompt": "   "})
    assert res["error"] == "bad_arguments"
    assert proc.scheduled == []


# ====================================================================== create_image_asset

@pytest.mark.asyncio
async def test_create_asset_without_a_container_generates_nothing():
    oc = _openai()
    res = await it.execute_create_image_asset(
        _ctx(_FakeProcessor(openai_client=oc), container_id=None),
        {"prompt": "a cover image", "filename": "cover.png"})

    assert res["ok"] is False and res["error"] == "sandbox_unavailable"
    assert "generate_image" in res["message"]        # the model is pointed at the right tool
    oc.generate_image.assert_not_awaited()           # and nothing was spent finding out


@pytest.mark.asyncio
@pytest.mark.critical
async def test_create_asset_mounts_the_bytes_and_posts_nothing():
    oc = _openai(create_path="/mnt/data/cover.png")
    proc = _FakeProcessor(openai_client=oc)
    client = _FakeClient()
    ctx = _ctx(proc, client, container_id="cntr_abc123")

    res = await it.execute_create_image_asset(
        ctx, {"prompt": "a cover image of a red cat", "filename": "cover.png"})

    assert res["ok"] is True and res["path"] == "/mnt/data/cover.png"
    assert "NOT been posted" in res["message"]

    # The bytes went INTO the container the model is actually running code in.
    create = oc.client.containers.files.create
    create.assert_awaited_once()
    assert create.await_args.kwargs["container_id"] == "cntr_abc123"
    assert create.await_args.kwargs["file"].name == "cover.png"

    # Recorded on the context, so a turn that publishes nothing can still rescue the image
    # rather than letting it die with the container.
    assert [a["path"] for a in ctx.sandbox_image_assets] == ["/mnt/data/cover.png"]
    assert ctx.sandbox_image_assets[0]["filename"] == "cover.png"

    # An INGREDIENT is not a deliverable: nothing reaches Slack from this tool.
    client.send_image.assert_not_awaited()
    client.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_asset_sanitizes_the_filename_the_model_chose():
    # The filename is untrusted input landing in a shell-adjacent sandbox path, and the
    # extension must match the bytes we actually made.
    oc = _openai()
    ctx = _ctx(_FakeProcessor(openai_client=oc), container_id="cntr_abc123",
               thread_config=_cfg(image_format="jpeg"))

    await it.execute_create_image_asset(
        ctx, {"prompt": "a cover", "filename": "../../etc/passwd"})

    assert oc.client.containers.files.create.await_args.kwargs["file"].name == "etc_passwd.jpg"


@pytest.mark.asyncio
async def test_create_asset_reports_a_failed_mount():
    oc = _openai(create_error=Exception("container is gone"))
    ctx = _ctx(_FakeProcessor(openai_client=oc), container_id="cntr_dead")

    res = await it.execute_create_image_asset(
        ctx, {"prompt": "a cover", "filename": "cover.png"})

    assert res["ok"] is False and res["error"] == "mount_failed"
    assert ctx.sandbox_image_assets == []     # nothing to rescue: there is no file


@pytest.mark.asyncio
async def test_create_asset_moderation_block_is_a_result_not_an_exception():
    oc = _openai(generate=AsyncMock(side_effect=Exception("moderation_blocked")))
    ctx = _ctx(_FakeProcessor(openai_client=oc), container_id="cntr_abc123")

    res = await it.execute_create_image_asset(
        ctx, {"prompt": "a famous logo", "filename": "logo.png"})

    assert res["ok"] is False and res["error"] == "moderation_blocked"
    assert "rephrased" in res["message"]      # the model is given a way forward
    oc.client.containers.files.create.assert_not_awaited()
    assert ctx.sandbox_image_assets == []


@pytest.mark.asyncio
async def test_create_asset_is_capped_per_turn():
    oc = _openai()
    ctx = _ctx(_FakeProcessor(openai_client=oc), container_id="cntr_abc123")
    ctx.sandbox_image_assets = [{"path": f"/mnt/data/{i}.png"} for i in range(4)]

    res = await it.execute_create_image_asset(
        ctx, {"prompt": "one more", "filename": "extra.png"})

    assert res["ok"] is False and res["error"] == "at_capacity"
    oc.generate_image.assert_not_awaited()


# ====================================================================== dispatch contract

@pytest.mark.asyncio
async def test_no_executor_raises_into_the_tool_loop():
    # The registry wraps failures, but a tool that raises loses its own error message — the
    # model then learns nothing about WHY. Dispatch the ugliest arguments through the real
    # registry and assert every result is a well-formed refusal.
    registry = ToolRegistry()
    it.register_image_tools(registry)
    ctx = _ctx(_FakeProcessor(), container_id=None, catalog=CATALOG)

    for name, args in [
        ("generate_image", "{}"),
        ("generate_image", "not json at all"),
        ("edit_image", json.dumps({"source_image_ids": "img_7", "prompt": "x"})),
        ("edit_image", json.dumps({"source_image_ids": ["img_nope"], "prompt": "x"})),
        ("create_image_asset", json.dumps({"prompt": "x", "filename": "y.png"})),
    ]:
        res = await registry.dispatch(ctx, name, args)
        assert res["ok"] is False, (name, res)
        assert res["error"] != "execution_error", (name, res)   # never an escaped exception
        assert res["message"]
