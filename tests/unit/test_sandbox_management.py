"""T4 — list_sandbox_files and reset_sandbox: seeing into the code sandbox, and replacing it.

Both tools are small, and both are shaped by one correction that cost a review round:

* LISTING must not CREATE. `ensure_sandbox()` would mint a container for a turn that started on
  `auto`, and listing a container made this instant proves only that a fresh container is empty.
  A turn with no sandbox is answered honestly instead.
* RESETTING cannot go through `invalidate` + `get_or_create`. `get_or_create` answers
  `{"type": "auto"}` for an unbound thread rather than creating anything, and the turn's SHARED
  `SandboxHolder` would go on handing every other tool the old id. So the binding is dropped, a
  container is created EXPLICITLY, and the holder is repointed — in that order, which is what the
  order tests below actually pin down.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor import file_mount
from message_processor.containers import AUTO_CONTAINER
from message_processor.tool_registry import SURFACE_CHANNEL, SandboxHolder, ToolContext, ToolRegistry


class _Pager:
    """containers.files.list() returns an async-iterable paginator, not a coroutine."""

    def __init__(self, files):
        self._files = list(files)

    def __aiter__(self):
        async def gen():
            for f in self._files:
                yield f
        return gen()


def _file(path, size=12, source="assistant"):
    return SimpleNamespace(id=f"cfile_{path.rsplit('/', 1)[-1]}", path=path, bytes=size,
                           source=source)


def _ctx(container="cntr_abc", files=(), holder=True, manager=None):
    raw = MagicMock()
    raw.containers.files.list = MagicMock(return_value=_Pager(files))
    processor = MagicMock()
    processor.openai_client.client = raw

    ctx = ToolContext(channel_id="C1", thread_ts="123.45", processor=processor,
                      client=MagicMock(), container_id=container)
    if holder:
        ctx.sandbox = SandboxHolder(container_id=container,
                                    manager=manager if manager is not None else MagicMock(),
                                    thread_key="C1:123.45")
    return ctx, raw


def _manager(created="cntr_new", calls=None):
    """A ContainerManager stand-in that records the ORDER of what the reset asks of it."""
    log = calls if calls is not None else []

    async def _invalidate(thread_key, container_id=None):
        log.append(("invalidate", thread_key, container_id))

    async def _create_explicit(thread_key):
        log.append(("create_explicit", thread_key))
        return created

    async def _get_or_create(thread_key):
        log.append(("get_or_create", thread_key))
        return AUTO_CONTAINER

    return MagicMock(invalidate=AsyncMock(side_effect=_invalidate),
                     create_explicit=AsyncMock(side_effect=_create_explicit),
                     get_or_create=AsyncMock(side_effect=_get_or_create),
                     bridge_container=AsyncMock(return_value=None)), log


# ============================================================================ schemas + gating

@pytest.mark.unit
class TestSchemas:
    def test_neither_tool_takes_an_argument(self):
        for schema in (file_mount.get_list_sandbox_files_schema(),
                       file_mount.get_reset_sandbox_schema()):
            assert schema["type"] == "function"
            assert schema["parameters"]["properties"] == {}
            assert "required" not in schema["parameters"]

    def test_the_listing_explains_the_source_field(self):
        # "who put this here" is the whole reason to show a listing rather than a path list.
        text = file_mount.get_list_sandbox_files_schema()["description"]
        assert "'user'" in text and "'assistant'" in text

    def test_reset_is_described_as_a_judgment_call_with_no_cadence(self):
        text = file_mount.get_reset_sandbox_schema()["description"]
        assert "poisoned" in text
        assert "PERMANENT" not in text          # it is not a deletion of anyone's work…
        assert "unreachable" in text            # …but the old sandbox does go
        # Owner ruling: never a routine, never a required pairing.
        assert "no cadence to keep" in text
        assert "no need to reset before or after any other tool" in text


@pytest.mark.unit
class TestRegistration:
    def _names(self, cfg, surface=None):
        registry = ToolRegistry()
        file_mount.register_file_mount_tools(registry)
        kwargs = {"surface": surface} if surface else {}
        return {s["name"] for s in registry.schemas(cfg, **kwargs)}

    def test_both_tools_on_both_surfaces_when_the_sandbox_is_on(self):
        cfg = {"enable_code_interpreter": True}
        for surface in (None, SURFACE_CHANNEL):
            names = self._names(cfg, surface=surface)
            assert {"list_sandbox_files", "reset_sandbox"} <= names

    def test_both_hidden_on_both_surfaces_when_the_sandbox_is_off(self):
        cfg = {"enable_code_interpreter": False}
        for surface in (None, SURFACE_CHANNEL):
            names = self._names(cfg, surface=surface)
            assert "list_sandbox_files" not in names
            assert "reset_sandbox" not in names

    def test_the_listing_needs_no_files_to_be_offered(self):
        # mount_file hides itself with an empty catalog; the sandbox tools are about the
        # CONTAINER, which exists (or does not) regardless of what the thread has shared.
        cfg = {"enable_code_interpreter": True, file_mount.FILES_KEY: []}
        assert "mount_file" not in self._names(cfg)
        assert "list_sandbox_files" in self._names(cfg)


# ================================================================================ listing

@pytest.mark.unit
class TestListSandboxFiles:
    async def test_reports_path_size_and_source_for_every_file(self):
        ctx, raw = _ctx(files=[_file("/mnt/data/chart.png", 900, "assistant"),
                               _file("/mnt/data/sales.csv", 12, "user")])

        result = await file_mount.execute_list_sandbox_files(ctx, {})

        assert result["ok"] is True
        assert result["container_id"] == "cntr_abc"
        assert result["count"] == 2
        assert result["files"] == [
            {"path": "/mnt/data/chart.png", "filename": "chart.png", "size_bytes": 900,
             "source": "assistant"},
            {"path": "/mnt/data/sales.csv", "filename": "sales.csv", "size_bytes": 12,
             "source": "user"},
        ]
        assert raw.containers.files.list.call_args.kwargs["container_id"] == "cntr_abc"

    async def test_an_unknown_size_is_None_rather_than_a_guess(self):
        ctx, _raw = _ctx(files=[_file("/mnt/data/deck.pptx", None)])
        result = await file_mount.execute_list_sandbox_files(ctx, {})
        assert result["files"][0]["size_bytes"] is None

    async def test_an_empty_container_says_so(self):
        ctx, _raw = _ctx(files=[])
        result = await file_mount.execute_list_sandbox_files(ctx, {})
        assert result["ok"] is True and result["files"] == []
        assert "empty" in result["message"]

    async def test_no_sandbox_is_an_honest_answer_not_an_error(self):
        ctx, raw = _ctx(container=None)
        ctx.sandbox = SandboxHolder(manager=MagicMock(), thread_key="C1:123.45")

        result = await file_mount.execute_list_sandbox_files(ctx, {})

        assert result["ok"] is True
        assert result["container_id"] is None and result["files"] == [] and result["count"] == 0
        assert "no code sandbox" in result["message"]
        raw.containers.files.list.assert_not_called()

    async def test_listing_never_mints_a_container(self):
        """The whole point of reading `sandbox_container_id()` rather than `ensure_sandbox()`:
        a container created to be listed is empty by construction."""
        manager = MagicMock(bridge_container=AsyncMock(return_value="cntr_would_be_made"))
        ctx, raw = _ctx(container=None)
        ctx.sandbox = SandboxHolder(manager=manager, thread_key="C1:123.45")

        result = await file_mount.execute_list_sandbox_files(ctx, {})

        assert result["container_id"] is None
        manager.bridge_container.assert_not_awaited()
        raw.containers.files.list.assert_not_called()

    async def test_a_recycled_container_is_refused_without_listing(self):
        ctx, raw = _ctx()
        ctx.container_gone_sink = ["cntr_abc"]

        result = await file_mount.execute_list_sandbox_files(ctx, {})

        assert result["ok"] is False and result["error"] == "container_recycled"
        raw.containers.files.list.assert_not_called()

    async def test_a_failing_listing_is_a_result_not_a_raise(self):
        ctx, raw = _ctx()
        raw.containers.files.list = MagicMock(side_effect=RuntimeError("expired"))

        result = await file_mount.execute_list_sandbox_files(ctx, {})

        assert result["ok"] is False and result["error"] == "listing_failed"

    async def test_no_processor_is_answered_honestly(self):
        ctx, _raw = _ctx()
        ctx.processor = None
        result = await file_mount.execute_list_sandbox_files(ctx, {})
        assert result["ok"] is False and result["error"] == "unavailable"


# ================================================================================== reset

@pytest.mark.unit
class TestResetSandbox:
    async def test_drops_the_binding_then_creates_and_repoints_the_holder(self):
        manager, calls = _manager(created="cntr_new")
        ctx, _raw = _ctx(container="cntr_old", manager=manager)

        result = await file_mount.execute_reset_sandbox(ctx, {})

        assert result["ok"] is True
        assert result["container_id"] == "cntr_new"
        assert result["previous_container_id"] == "cntr_old"
        # ORDER is the correction: create_explicit REUSES a live binding, so a create that ran
        # first would hand back the very container being reset.
        assert calls == [("invalidate", "C1:123.45", "cntr_old"),
                         ("create_explicit", "C1:123.45")]
        # The SHARED holder now names the replacement — this is what the tool loop pins into the
        # next round's declaration and what every sibling tool call resolves.
        assert ctx.sandbox.container_id == "cntr_new"
        assert ctx.sandbox_container_id() == "cntr_new"

    async def test_get_or_create_is_never_the_path(self):
        """It answers {"type": "auto"} for an unbound thread — no container, nothing to hand
        back, and the holder would keep serving the old id."""
        manager, _calls = _manager()
        ctx, _raw = _ctx(container="cntr_old", manager=manager)

        await file_mount.execute_reset_sandbox(ctx, {})

        manager.get_or_create.assert_not_awaited()

    async def test_the_new_container_is_what_later_tools_mount_into(self):
        manager, _calls = _manager(created="cntr_new")
        ctx, _raw = _ctx(container="cntr_old", manager=manager)

        await file_mount.execute_reset_sandbox(ctx, {})

        assert await ctx.ensure_sandbox() == "cntr_new"

    async def test_a_turn_with_no_container_yet_still_gets_a_clean_one(self):
        manager, calls = _manager(created="cntr_new")
        ctx, _raw = _ctx(container=None, manager=manager)

        result = await file_mount.execute_reset_sandbox(ctx, {})

        assert result["ok"] is True and result["previous_container_id"] is None
        assert calls[0] == ("invalidate", "C1:123.45", None)

    async def test_an_auto_container_answer_is_not_a_reset(self):
        """AUTO_CONTAINER is a dict — a throwaway the API may put the model in, but not an
        addressable id we can hand back or push bytes into. The holder must not be repointed at
        it, or the turn would believe it had a sandbox it cannot name."""
        manager, _calls = _manager()
        manager.create_explicit = AsyncMock(return_value=AUTO_CONTAINER)
        ctx, _raw = _ctx(container="cntr_old", manager=manager)

        result = await file_mount.execute_reset_sandbox(ctx, {})

        assert result["ok"] is False and result["error"] == "reset_failed"
        assert ctx.sandbox.container_id == "cntr_old"

    async def test_getting_the_same_container_back_is_a_failure_not_a_reset(self):
        """`invalidate` swallows a durable-delete failure by contract, and a binding that
        survived is one `create_explicit` rediscovers as still alive and hands straight back.
        Reporting success there would send the model into the very sandbox it asked to escape,
        believing it was clean."""
        manager, _calls = _manager(created="cntr_old")   # the drop did not take
        ctx, _raw = _ctx(container="cntr_old", manager=manager)

        result = await file_mount.execute_reset_sandbox(ctx, {})

        assert result["ok"] is False and result["error"] == "reset_failed"
        assert "nothing was cleared" in result["message"]
        assert ctx.sandbox.container_id == "cntr_old"

    async def test_a_first_container_for_an_auto_turn_is_not_mistaken_for_that(self):
        # No previous id at all, so "the same one came back" cannot apply.
        manager, _calls = _manager(created="cntr_new")
        ctx, _raw = _ctx(container=None, manager=manager)

        result = await file_mount.execute_reset_sandbox(ctx, {})

        assert result["ok"] is True and ctx.sandbox.container_id == "cntr_new"

    async def test_a_failing_create_leaves_the_holder_alone(self):
        manager, _calls = _manager()
        manager.create_explicit = AsyncMock(side_effect=RuntimeError("no capacity"))
        ctx, _raw = _ctx(container="cntr_old", manager=manager)

        result = await file_mount.execute_reset_sandbox(ctx, {})

        assert result["ok"] is False and result["error"] == "reset_failed"
        assert ctx.sandbox.container_id == "cntr_old"

    async def test_no_holder_means_nothing_to_replace(self):
        # A background job's hand-built context owns its container elsewhere; swapping an id we
        # do not hold would strand the caller on the old one silently.
        ctx, _raw = _ctx(container="cntr_old", holder=False)

        result = await file_mount.execute_reset_sandbox(ctx, {})

        assert result["ok"] is False and result["error"] == "unavailable"

    async def test_no_manager_or_thread_key_means_nothing_to_create_with(self):
        ctx, _raw = _ctx(container="cntr_old")
        ctx.sandbox = SandboxHolder(container_id="cntr_old", manager=None,
                                    thread_key="C1:123.45")
        assert (await file_mount.execute_reset_sandbox(ctx, {}))["error"] == "unavailable"

        manager, _calls = _manager()
        ctx.sandbox = SandboxHolder(container_id="cntr_old", manager=manager, thread_key=None)
        result = await file_mount.execute_reset_sandbox(ctx, {})
        assert result["error"] == "unavailable"
        manager.create_explicit.assert_not_awaited()

    async def test_the_work_claim_is_staked_before_the_create(self):
        manager, _calls = _manager()
        ctx, _raw = _ctx(container="cntr_old", manager=manager)
        ctx.turn = MagicMock(claim_work=AsyncMock())

        await file_mount.execute_reset_sandbox(ctx, {})

        ctx.turn.claim_work.assert_awaited_once()

    async def test_a_broken_work_claim_never_breaks_the_reset(self):
        manager, _calls = _manager(created="cntr_new")
        ctx, _raw = _ctx(container="cntr_old", manager=manager)
        ctx.turn = MagicMock(claim_work=AsyncMock(side_effect=RuntimeError("no message")))

        result = await file_mount.execute_reset_sandbox(ctx, {})

        assert result["ok"] is True and ctx.sandbox.container_id == "cntr_new"


@pytest.mark.unit
class TestHolderReplace:
    def test_replace_swaps_the_shared_id(self):
        holder = SandboxHolder(container_id="cntr_old")
        holder.replace("cntr_new")
        assert holder.container_id == "cntr_new"

    def test_replace_with_nothing_empties_it_rather_than_storing_a_blank(self):
        holder = SandboxHolder(container_id="cntr_old")
        holder.replace("")
        assert holder.container_id is None
