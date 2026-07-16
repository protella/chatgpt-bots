"""F36 — canvas tools: create, read, edit, list."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor import canvas_tools as ct
from tool_registry import ToolContext, ToolRegistry


def _tabs(*canvas_ids):
    """conversations.info properties: a canvas tab per id, plus the furniture Slack always sends."""
    tabs = [{"type": "bookmarks", "id": "bookmarks"}]
    tabs += [{"type": "canvas", "id": f"Ct{i}", "data": {"file_id": fid}}
             for i, fid in enumerate(canvas_ids)]
    tabs.append({"type": "files", "id": "files", "is_disabled": True})
    return {"channel": {"properties": {"canvas": None, "tabs": tabs}}}


def _ctx(*, channels=("C1",), groups=(), filetype="quip", channel_canvas=None):
    web = MagicMock()
    web.canvases_create = AsyncMock(return_value={"canvas_id": "F123"})
    web.conversations_canvases_create = AsyncMock(return_value={"canvas_id": "F123"})
    web.canvases_access_set = AsyncMock(return_value={"ok": True})
    web.canvases_edit = AsyncMock(return_value={"ok": True})
    web.canvases_delete = AsyncMock(return_value={"ok": True})
    web.canvases_sections_lookup = AsyncMock(
        return_value={"sections": [{"id": "temp:C:abc"}]})
    web.files_info = AsyncMock(return_value={
        "file": {"id": "F123", "filetype": filetype, "channels": list(channels),
                 "groups": list(groups),
                 "permalink": "https://slack.com/docs/F123",
                 "url_private": "https://files.slack.com/canvas"}})
    web.files_list = AsyncMock(return_value={
        "files": [{"id": "F123", "title": "Launch plan", "updated": 1}]})
    # By default the channel has NO channel canvas — the state in which create is on offer.
    web.conversations_info = AsyncMock(
        return_value=_tabs(*( [channel_canvas] if channel_canvas else [] )))

    client = MagicMock()
    client.app = MagicMock()
    client.app.client = web
    client.download_file = AsyncMock(
        return_value=b'<div class="quip-canvas-content"><h1>Plan</h1>'
                     b'<p>Ship on <strong>Friday</strong>.</p>'
                     b'<ul><li>one</li><li>two</li></ul></div>')

    return ToolContext(channel_id="C1", thread_ts="1.0", client=client), web


@pytest.mark.unit
class TestCreate:
    async def test_creates_the_channel_canvas_not_a_standalone_one(self):
        # THE distinction. Only conversations.canvases.create yields a canvas Slack pins as a tab;
        # a standalone canvas (canvases.create) cannot be pinned at all, and an unpinned canvas
        # appears nowhere in the channel.
        ctx, web = _ctx()
        out = await ct.execute_create_channel_canvas(ctx, {"title": "Plan", "markdown": "Ship it"})

        assert out["ok"] is True
        assert out["canvas_id"] == "F123"
        assert out["is_channel_canvas"] is True
        web.conversations_canvases_create.assert_awaited_once()
        assert web.conversations_canvases_create.call_args.kwargs["channel_id"] == "C1"
        web.canvases_create.assert_not_awaited()
        # The channel canvas belongs to the channel; Slack shares it in on creation.
        web.canvases_access_set.assert_not_awaited()

    async def test_the_title_is_passed_to_slack_and_labels_the_tab(self):
        """`title` is UNDOCUMENTED and is not in the slack_sdk signature — which takes **kwargs,
        so it reaches the API anyway, and it works: files.info reports it and the canvas TAB is
        labelled with it. Reading the signature and concluding "no title is possible" shipped a
        canvas called "Untitled", which is a document no ask can ever match. It is also the ONLY
        chance to set one — there is no rename API of any kind.
        """
        ctx, web = _ctx()
        await ct.execute_create_channel_canvas(ctx, {"title": "DevOps Agenda",
                                                     "markdown": "First item"})
        kwargs = web.conversations_canvases_create.call_args.kwargs
        assert kwargs["title"] == "DevOps Agenda"
        # ...and the body is left exactly as written — no synthesised heading duplicating the
        # title Slack now renders above it.
        assert kwargs["document_content"]["markdown"] == "First item"

    async def test_a_second_channel_canvas_is_refused(self):
        # conversations.canvases.create is NOT idempotent: a second call means a second canvas
        # and a second tab, permanently. The schema hides the tool once one exists, but the
        # catalog behind that schema can be stale, so the executor checks live too.
        ctx, web = _ctx(channel_canvas="F123")
        out = await ct.execute_create_channel_canvas(ctx, {"title": "T", "markdown": "x"})

        assert out["ok"] is False
        assert out["error"] == "already_exists"
        assert out["canvas_id"] == "F123"
        web.conversations_canvases_create.assert_not_awaited()

    async def test_a_tab_pointing_at_a_dead_canvas_does_not_block_creation(self):
        # A tab outlives its canvas by a while, so the tab alone is not proof one exists. Absence
        # from files.list is not proof of death either (it lags a fresh canvas), so the deciding
        # question is put to files.info, which answers `file_deleted`.
        ctx, web = _ctx(channel_canvas="F_GONE")
        web.files_info = AsyncMock(side_effect=Exception(
            "The server responded with: {'ok': False, 'error': 'file_deleted'}"))

        out = await ct.execute_create_channel_canvas(ctx, {"title": "T", "markdown": "x"})

        assert out["ok"] is True
        web.conversations_canvases_create.assert_awaited_once()

    async def test_empty_content_refused(self):
        ctx, _ = _ctx()
        out = await ct.execute_create_channel_canvas(ctx, {"title": "", "markdown": ""})
        assert out["ok"] is False

    async def test_oversize_refused(self):
        ctx, web = _ctx()
        out = await ct.execute_create_channel_canvas(
            ctx, {"title": "T", "markdown": "x" * (ct.MAX_MARKDOWN_CHARS + 1)})
        assert out["error"] == "too_long"
        web.conversations_canvases_create.assert_not_awaited()

    async def test_a_refusal_from_slack_is_reported_honestly(self):
        ctx, web = _ctx()
        web.conversations_canvases_create = AsyncMock(side_effect=RuntimeError("no scope"))
        out = await ct.execute_create_channel_canvas(ctx, {"title": "T", "markdown": "x"})
        assert out["ok"] is False
        assert out["error"] == "create_failed"


@pytest.mark.unit
class TestAuthorization:
    """A canvas id is just a Slack file id. One from ANOTHER channel is syntactically perfect
    and would edit happily — and silently rewriting someone else's document is not a failure a
    user ever sees coming."""

    async def test_editing_a_canvas_from_another_channel_is_refused(self):
        ctx, web = _ctx(channels=("C_SOMEWHERE_ELSE",))
        out = await ct.execute_edit_canvas(
            ctx, {"canvas_id": "F123", "operation": "append", "markdown": "hi"})

        assert out["error"] == "not_in_this_channel"
        web.canvases_edit.assert_not_awaited()

    async def test_reading_a_canvas_from_another_channel_is_refused(self):
        ctx, _ = _ctx(channels=("C_OTHER",))
        out = await ct.execute_read_canvas(ctx, {"canvas_id": "F123"})
        assert out["error"] == "not_in_this_channel"

    async def test_a_non_canvas_file_is_refused(self):
        # An id for a PDF in this channel is still not a canvas.
        ctx, web = _ctx(filetype="pdf")
        out = await ct.execute_edit_canvas(
            ctx, {"canvas_id": "F123", "operation": "append", "markdown": "hi"})
        assert out["error"] == "not_in_this_channel"
        web.canvases_edit.assert_not_awaited()


@pytest.mark.unit
class TestEdit:
    async def test_append_and_prepend_map_to_the_right_operations(self):
        ctx, web = _ctx()
        await ct.execute_edit_canvas(
            ctx, {"canvas_id": "F123", "operation": "append", "markdown": "more"})
        assert web.canvases_edit.call_args.kwargs["changes"][0]["operation"] == "insert_at_end"

        await ct.execute_edit_canvas(
            ctx, {"canvas_id": "F123", "operation": "prepend", "markdown": "first"})
        assert web.canvases_edit.call_args.kwargs["changes"][0]["operation"] == "insert_at_start"

    async def test_replace_section_targets_the_matched_passage(self):
        ctx, web = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_section",
            "find_text": "Ship on Friday", "markdown": "Ship on Monday"})

        assert out["ok"] is True
        change = web.canvases_edit.call_args.kwargs["changes"][0]
        assert change["operation"] == "replace"
        assert change["section_id"] == "temp:C:abc"

    async def test_replace_section_without_find_text_is_refused(self):
        ctx, web = _ctx()
        out = await ct.execute_edit_canvas(
            ctx, {"canvas_id": "F123", "operation": "replace_section", "markdown": "x"})
        assert out["error"] == "missing_find_text"
        web.canvases_edit.assert_not_awaited()

    async def test_an_ambiguous_match_refuses_rather_than_guessing(self):
        # Two matches means a coin flip over which of the user's paragraphs gets overwritten.
        ctx, web = _ctx()
        web.canvases_sections_lookup = AsyncMock(
            return_value={"sections": [{"id": "a"}, {"id": "b"}]})
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_section",
            "find_text": "the", "markdown": "x"})

        assert out["error"] == "ambiguous_find_text"
        assert out["matches"] == 2
        web.canvases_edit.assert_not_awaited()

    async def test_no_match_says_so(self):
        ctx, web = _ctx()
        web.canvases_sections_lookup = AsyncMock(return_value={"sections": []})
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_section",
            "find_text": "nowhere", "markdown": "x"})
        assert out["error"] == "section_not_found"
        web.canvases_edit.assert_not_awaited()


@pytest.mark.unit
class TestRead:
    async def test_canvas_html_converts_back_to_markdown(self):
        # There is no canvases.read: content comes back as HTML from url_private, so the round
        # trip only works if we can turn it back into the markdown the model wrote.
        ctx, _ = _ctx()
        out = await ct.execute_read_canvas(ctx, {"canvas_id": "F123"})

        assert out["ok"] is True
        md = out["markdown"]
        assert "# Plan" in md
        assert "**Friday**" in md
        assert "- one" in md and "- two" in md


@pytest.mark.unit
class TestHtmlToMarkdown:
    def test_headings_lists_and_inline_marks(self):
        html = ('<div class="quip-canvas-content">'
                '<h2>Status</h2><p>We are <em>close</em> to <code>done</code>.</p>'
                '<ol><li>alpha</li><li>beta</li></ol>'
                '<p><a href="https://x.com">link</a></p></div>')
        md = ct._html_to_markdown(html)
        assert "## Status" in md
        assert "_close_" in md and "`done`" in md
        assert "1. alpha" in md and "2. beta" in md
        assert "[link](https://x.com)" in md

    def test_empty_is_empty_not_an_error(self):
        assert ct._html_to_markdown("") == ""

    def test_a_checklist_reads_back_as_a_checklist(self):
        """The signal is the LIST's style (`data-section-style='7'`), not the item: a ticked item
        is `<li class='checked'>` and an unticked one carries NO marker at all. Keying off the
        item made an untouched checklist read back as plain bullets — so the model could not tell
        a checklist from a bullet list, nor see what was already done."""
        html = ('<div class="quip-canvas-content">'
                "<div data-section-style='7'><ul>"
                "<li>alpha</li>"
                "<li class='checked'>beta</li>"
                "</ul></div></div>")
        md = ct._html_to_markdown(html)
        assert "- [ ] alpha" in md
        assert "- [x] beta" in md

    def test_bullet_and_numbered_lists_keep_their_kind(self):
        # Both are <ul> in canvas HTML — only the style says which is which.
        bullets = ct._html_to_markdown(
            '<div class="quip-canvas-content">'
            "<div data-section-style='5'><ul><li>one</li><li>two</li></ul></div></div>")
        numbered = ct._html_to_markdown(
            '<div class="quip-canvas-content">'
            "<div data-section-style='6'><ul><li>one</li><li>two</li></ul></div></div>")
        assert "- one" in bullets and "1." not in bullets
        assert "1. one" in numbered and "2. two" in numbered

    def test_tables_survive_the_round_trip(self):
        # Table cells hold <p>, so walking every <p> in the document shredded a table into a run
        # of loose paragraphs — the model then "reformatted" it back as bullets.
        html = ('<div class="quip-canvas-content"><table>'
                "<tr><td><p>Owner</p></td><td><p>Status</p></td></tr>"
                "<tr><td><p>Dana</p></td><td><p>Done</p></td></tr>"
                "</table></div>")
        md = ct._html_to_markdown(html)
        assert "| Owner | Status |" in md
        assert "| --- | --- |" in md
        assert "| Dana | Done |" in md

    def test_links_come_back_as_lnk_not_a(self):
        # Slack canvases emit <lnk href=…>. Reading only <a> dropped every link silently.
        md = ct._html_to_markdown(
            '<div class="quip-canvas-content">'
            '<p>see <lnk href="https://x.com">the doc</lnk></p></div>')
        assert "[the doc](https://x.com)" in md


@pytest.mark.unit
class TestRegistration:
    def test_addressed_turns_get_the_full_set_including_delete(self):
        registry = ToolRegistry()
        ct.register_canvas_tools(registry)
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "Agenda"}],
               "_canvas_delete_authorized": True}
        names = {s["name"] for s in registry.schemas(cfg)}

        assert names == {"create_channel_canvas", "list_canvases", "read_canvas", "edit_canvas",
                         "delete_canvas"}

    def test_an_unaddressed_turn_gets_everything_except_delete(self):
        # It can still create, read and edit while listening — it just cannot destroy anything
        # on its own initiative. A config with no authorization flag (or an explicit False) fails
        # CLOSED: delete is a destructive op and must not default to available.
        registry = ToolRegistry()
        ct.register_canvas_tools(registry)
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "Agenda"}],
               "_canvas_delete_authorized": False}
        names = {s["name"] for s in registry.schemas(cfg)}

        assert names == {"create_channel_canvas", "list_canvases", "read_canvas", "edit_canvas"}
        # And the same holds when the flag is simply absent — fail closed, not open.
        assert "delete_canvas" not in {
            s["name"] for s in registry.schemas({ct.CATALOG_KEY: cfg[ct.CATALOG_KEY]})}

    def test_with_no_canvases_only_create_and_list_are_offered(self):
        # read/edit are factories over the channel's canvases: with none, there is nothing to
        # name, and offering them would only invite an invented id.
        registry = ToolRegistry()
        ct.register_canvas_tools(registry)
        names = {s["name"] for s in registry.schemas({})}

        assert names == {"create_channel_canvas", "list_canvases"}

    def test_create_disappears_once_the_channel_canvas_exists(self):
        # "Create if not exists" is not a rule the model has to remember — it is the shape of the
        # toolset. A second conversations.canvases.create would mean a second permanent tab.
        registry = ToolRegistry()
        ct.register_canvas_tools(registry)
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "Agenda",
                                 "is_channel_canvas": True}]}
        names = {s["name"] for s in registry.schemas(cfg)}

        assert "create_channel_canvas" not in names
        assert {"read_canvas", "edit_canvas"} <= names

    def test_the_channel_canvas_is_not_deletable(self):
        # It is the channel's own document and its tab is furniture. "Clear the agenda" is an
        # edit, never a delete — so it is not even in the enum.
        registry = ToolRegistry()
        ct.register_canvas_tools(registry)
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "Agenda",
                                 "is_channel_canvas": True}]}
        delete = [s for s in registry.schemas(cfg) if s["name"] == "delete_canvas"]

        assert delete == []       # nothing else to offer, so the tool is withheld entirely

    def test_delete_still_offered_for_a_standalone_canvas_beside_it(self):
        registry = ToolRegistry()
        ct.register_canvas_tools(registry)
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "Agenda",
                                 "is_channel_canvas": True},
                                {"canvas_id": "F2", "title": "Old notes"}],
               "_canvas_delete_authorized": True}
        delete = [s for s in registry.schemas(cfg) if s["name"] == "delete_canvas"][0]

        assert delete["parameters"]["properties"]["canvas_id"]["enum"] == ["F2"]


@pytest.mark.unit
class TestPrivateChannels:
    """Slack reports a file's channels in THREE places depending on the channel type:
    `channels` for public, `groups` for PRIVATE, `ims` for DMs. Checking only `channels` reads a
    private channel as "not shared here" and refuses every edit — which is exactly what happened
    live, in a private channel, where `channels` came back empty and `groups` held the id."""

    async def test_a_canvas_in_a_private_channel_is_editable(self):
        ctx, web = _ctx(channels=(), groups=("C1",))
        out = await ct.execute_edit_canvas(
            ctx, {"canvas_id": "F123", "operation": "append", "markdown": "more"})

        assert out["ok"] is True
        web.canvases_edit.assert_awaited_once()

    async def test_a_private_channel_elsewhere_is_still_refused(self):
        ctx, web = _ctx(channels=(), groups=("C_OTHER",))
        out = await ct.execute_edit_canvas(
            ctx, {"canvas_id": "F123", "operation": "append", "markdown": "more"})

        assert out["error"] == "not_in_this_channel"
        web.canvases_edit.assert_not_awaited()


@pytest.mark.unit
class TestCanvasHtmlIsNotALoginPage:
    """`download_file` normally REFUSES an html body: for an image, html means auth failed and
    Slack served a login page rather than a 401. A canvas's content genuinely is html, so it
    opts out of that guard — and therefore has to catch the login page itself."""

    async def test_read_opts_out_of_the_html_guard(self):
        ctx, _ = _ctx()
        await ct.execute_read_canvas(ctx, {"canvas_id": "F123"})
        assert ctx.client.download_file.call_args.kwargs["allow_html"] is True

    async def test_a_login_page_is_not_mistaken_for_a_canvas(self):
        ctx, _ = _ctx()
        ctx.client.download_file = AsyncMock(
            return_value=b"<html><body><h1>Sign in to Slack</h1></body></html>")
        out = await ct.execute_read_canvas(ctx, {"canvas_id": "F123"})
        assert out["ok"] is False
        assert out["error"] == "read_failed"


@pytest.mark.unit
class TestFindTextRoundTrip:
    """`read_canvas` renders the canvas as MARKDOWN, so a list item comes back as
    `- Launch lead — Dana`. The model quotes exactly that as find_text — but Slack searches the
    canvas TEXT, where the bullet is structure, not content. The lookup missed and the edit was
    refused. Verified live: with the bullet, section_not_found; without it, matched first time.
    """

    @pytest.mark.parametrize("quoted,searched", [
        ("- Launch lead — Dana", "Launch lead — Dana"),
        ("## Key dates", "Key dates"),
        ("1. alpha", "alpha"),
        ("- [ ] ship it", "ship it"),
        ("**Launch date** — Nov 3", "Launch date — Nov 3"),
    ])
    def test_markdown_scaffolding_is_stripped(self, quoted, searched):
        assert ct._searchable(quoted) == searched

    async def test_the_lookup_searches_the_plain_text(self):
        ctx, web = _ctx()
        await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_section",
            "find_text": "- Launch lead — TBD", "markdown": "- Launch lead — Dana"})

        sent = web.canvases_sections_lookup.call_args.kwargs["criteria"]["contains_text"]
        assert sent == "Launch lead — TBD"

    async def test_find_text_that_is_only_scaffolding_is_refused(self):
        ctx, web = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_section",
            "find_text": "- ", "markdown": "x"})
        assert out["error"] == "missing_find_text"
        web.canvases_sections_lookup.assert_not_awaited()


@pytest.mark.unit
class TestSectionIsOneBlock:
    """A canvas section is ONE block — a heading, a paragraph, a single list item. It is not a
    region. The model reaches for a region anyway: it reads the canvas, sees a Steps list, and
    quotes the whole list to "replace the Steps section". That can never match, so the tool has
    to name the unit rather than refuse with a not-found it cannot learn from."""

    async def test_a_multi_line_find_text_is_refused_with_a_usable_reason(self):
        ctx, web = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_section",
            "find_text": "Steps\n- Step 1\n- Step 2", "markdown": "x"})

        assert out["error"] == "find_text_spans_multiple_lines"
        # The message must tell it what to do instead, not merely that it failed.
        assert "one per line" in out["message"] or "once per line" in out["message"]
        web.canvases_sections_lookup.assert_not_awaited()

    async def test_a_single_list_item_is_fine(self):
        ctx, web = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_section",
            "find_text": "- Step 1", "markdown": "Step 1 — done, owner Dana"})

        assert out["ok"] is True
        assert web.canvases_sections_lookup.call_args.kwargs["criteria"]["contains_text"] == "Step 1"


@pytest.mark.unit
class TestReplacementStaysInTheList:
    """A section IS the list item, so its replacement is the item's CONTENT. Hand Slack a bullet
    and it parses a whole new list — verified live: replacing "beta" with `- [x] beta` deleted it
    from the list and appended a fresh one-item list at the bottom of the canvas. Without the
    bullet it swaps in place, keeping the item's own id.

    A CHECKBOX is the exception, and stripping is not good enough for it: `- [x] beta` reduced to
    `beta` lands in place with its tick UNCHANGED — a "mark it done" that reports success and does
    nothing. So a checkbox replacement is refused outright and routed to replace_list (see
    TestChecklists); these cases are the ones where stripping is genuinely right.
    """

    @pytest.mark.parametrize("given,sent", [
        ("- Owner: Dana", "Owner: Dana"),
        ("1. alpha", "alpha"),
        ("plain text", "plain text"),
        # A heading's ## must SURVIVE — strip it and the heading silently demotes to a paragraph.
        ("## Key dates", "## Key dates"),
    ])
    def test_list_bullets_are_stripped_but_headings_are_not(self, given, sent):
        assert ct._replacement_for_section(given) == sent

    async def test_the_edit_sends_the_normalised_replacement(self):
        ctx, web = _ctx()
        await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_section",
            "find_text": "Owner", "markdown": "- Owner: Dana"})

        sent = web.canvases_edit.call_args.kwargs["changes"][0]["document_content"]["markdown"]
        assert sent == "Owner: Dana"

    async def test_append_is_left_alone(self):
        # Only replace_section has the in-place constraint; an append genuinely wants its bullet.
        ctx, web = _ctx()
        await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "append", "markdown": "- a new bullet"})

        sent = web.canvases_edit.call_args.kwargs["changes"][0]["document_content"]["markdown"]
        assert sent == "- a new bullet"


@pytest.mark.unit
class TestTheModelKnowsCanvasesExist:
    """Slack posts NO message when a canvas is shared, so canvases never appear in the rebuilt
    history. Before this, the only signal a canvas existed was the word "canvas" in a tool
    description — so "update our devops call agenda to discuss failed deploys" had nothing to
    match on, and the model would have had to guess that a canvas might exist and call
    list_canvases on a hunch. Same fix as mount_file/edit_image: put the ids in front of it.
    """

    def setup_method(self):
        ct._catalog_cache.clear()

    def test_the_ids_and_titles_ride_in_the_schema(self):
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "DevOps Agenda"},
                                {"canvas_id": "F2", "title": "Q4 Launch Plan"}]}
        schema = ct.get_edit_canvas_schema(cfg)

        assert schema["parameters"]["properties"]["canvas_id"]["enum"] == ["F1", "F2"]
        # The TITLE is what "our devops call agenda" actually matches against.
        assert "DevOps Agenda" in schema["description"]
        assert "Q4 Launch Plan" in schema["description"]

    def test_read_carries_the_catalog_too(self):
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "DevOps Agenda"}]}
        schema = ct.get_read_canvas_schema(cfg)
        assert schema["parameters"]["properties"]["canvas_id"]["enum"] == ["F1"]
        assert "DevOps Agenda" in schema["description"]

    def test_no_canvases_means_the_tools_are_not_offered(self):
        # Nothing to read or edit: hiding them beats inviting an invented id.
        assert ct.get_edit_canvas_schema({ct.CATALOG_KEY: []}) is None
        assert ct.get_read_canvas_schema(None) is None

    async def test_catalog_is_cached_per_channel(self):
        # This one is a Slack API call, not a DB read — it must not run on every turn.
        client = MagicMock()
        web = MagicMock()
        web.files_list = AsyncMock(return_value={"files": [{"id": "F1", "title": "Agenda"}]})
        web.conversations_info = AsyncMock(return_value=_tabs())
        client.app = MagicMock()
        client.app.client = web

        a = await ct.build_catalog(client, "C1", now=100.0)
        b = await ct.build_catalog(client, "C1", now=200.0)     # inside the TTL
        assert a == b == [{"canvas_id": "F1", "title": "Agenda", "is_channel_canvas": False}]
        assert web.files_list.await_count == 1

        await ct.build_catalog(client, "C1", now=100.0 + ct._CATALOG_TTL + 1)  # expired
        assert web.files_list.await_count == 2

    async def test_the_channel_canvas_is_marked_and_named_by_its_heading(self):
        # Slack reports the channel canvas as "Untitled" and will not let us rename it, so the
        # document's own top heading is the only name it has — and a document called "Untitled"
        # is one no ask can ever match.
        client = MagicMock()
        web = MagicMock()
        web.files_list = AsyncMock(return_value={
            "files": [{"id": "F1", "title": "Untitled"}, {"id": "F2", "title": "Old notes"}]})
        web.conversations_info = AsyncMock(return_value=_tabs("F1"))
        web.files_info = AsyncMock(return_value={
            "file": {"url_private": "https://files.slack.com/canvas"}})
        client.app = MagicMock()
        client.app.client = web
        client.download_file = AsyncMock(
            return_value=b'<div class="quip-canvas-content"><h1>DevOps Agenda</h1></div>')

        entries = await ct.build_catalog(client, "C1", now=1.0)

        assert entries[0] == {"canvas_id": "F1", "title": "DevOps Agenda",
                              "is_channel_canvas": True}
        assert entries[1]["is_channel_canvas"] is False
        assert ct.channel_canvas_id(entries) == "F1"
        # ...and it is offered to the model BY ROLE, since "the canvas" means this one.
        assert "the channel canvas" in ct.catalog_lines(entries)
        assert "DevOps Agenda" in ct.catalog_lines(entries)

    async def test_a_stale_tab_is_not_mistaken_for_a_channel_canvas(self):
        # A tab outlives the canvas it points at. Trusting it would hand back a dead id — but
        # only files.info can say so (files.list lags in both directions).
        client = MagicMock()
        web = MagicMock()
        web.files_list = AsyncMock(return_value={"files": [{"id": "F1", "title": "Agenda"}]})
        web.conversations_info = AsyncMock(return_value=_tabs("F_DELETED"))
        web.files_info = AsyncMock(side_effect=Exception(
            "The server responded with: {'ok': False, 'error': 'file_deleted'}"))
        client.app = MagicMock()
        client.app.client = web

        entries = await ct.build_catalog(client, "C1", now=1.0)
        assert ct.channel_canvas_id(entries) is None

    async def test_creating_a_canvas_invalidates_the_cache(self):
        # Otherwise the turn that just made a canvas could not then edit it.
        ct._catalog_cache["C1"] = {"at": 0.0, "entries": []}
        ctx, _ = _ctx()
        await ct.execute_create_channel_canvas(ctx, {"title": "T", "markdown": "x"})
        assert "C1" not in ct._catalog_cache


@pytest.mark.unit
class TestCanvasesAreChannelContext:
    """Canvases are channel furniture, like the topic or the member list.

    Slack posts NO message when a canvas is shared, so a canvas is otherwise invisible: nothing
    in the rebuilt history mentions it. Two consumers need to know, and only one of them can see
    tool schemas:
      * the main model, so "update our devops call agenda" attaches to something;
      * the PARTICIPATION GATE, which sees no tools at all — without this a passive "we should
        update the devops agenda" reads as idle chatter and the bot stays silent, so the main
        model never gets a turn to notice the canvas in the first place.
    """

    def test_canvas_titles_land_in_the_channel_context_block(self):
        import inspect
        from message_processor.utilities import MessageUtilitiesMixin
        src = inspect.getsource(MessageUtilitiesMixin._get_system_prompt)
        assert 'channel_info.get("canvases")' in src
        assert "Channel canvases" in src

    def test_the_prompt_says_a_living_document_is_a_canvas(self):
        """Found live: asked to "start a running agenda for our devops call", the bot wrote the
        agenda as a CHAT MESSAGE, with create_channel_canvas sitting unused in its 23 tools. A
        tool description is read only AFTER the model decides to reach for a tool; the choice
        between "reply" and "document" happens before that, and only the system prompt shapes it.
        """
        import inspect
        from prompts import CANVAS_GUIDANCE
        from message_processor.utilities import MessageUtilitiesMixin

        flat = " ".join(CANVAS_GUIDANCE.lower().split())
        assert "start an agenda" in flat
        assert "that is a canvas, not a chat message" in flat
        # Also found live, on the very next run: told to start a devops agenda, it read the
        # channel's UNRELATED "Q4 Launch — Hot Sauce" canvas and tried to append the agenda to
        # it. Only Slack's ACL (restricted_action — the bot didn't create that one) stopped a
        # silent rewrite of someone else's document. "The only canvas here" is not "the canvas
        # they meant".
        assert "never write what was asked for into an unrelated canvas" in flat

        src = inspect.getsource(MessageUtilitiesMixin._get_system_prompt)
        assert "canvas_context" in src
        # ...and it rides the prompt only in a CHANNEL, where a canvas can actually exist.
        assert "channel_info is not None and config.enable_canvas_tools" in src

    def test_the_participation_gate_is_told_too(self):
        import inspect
        from openai_client.api import responses
        src = inspect.getsource(responses)
        assert 'signals.get("channel_canvases")' in src
        assert "Channel canvases" in src

        from message_processor.participation import ParticipationEngine
        sig = inspect.signature(ParticipationEngine.evaluate)
        assert "channel_canvases" in sig.parameters


@pytest.mark.unit
class TestDelete:
    """Deleting a canvas is irreversible and public. The danger was never the user asking —
    "delete that canvas" is an ordinary request and they own their channel. The danger is the
    bot deciding to TIDY UP: on an unprompted turn it is reading along in a channel nobody
    addressed it in, inferring what would be helpful. A tool that destroys a shared document has
    no business being on the table for that."""

    def test_offered_when_a_person_addressed_the_bot(self):
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "Old Agenda"}],
               "_canvas_delete_authorized": True}
        assert ct._delete_enabled(cfg) is True
        schema = ct.get_delete_canvas_schema(cfg)
        assert schema["parameters"]["properties"]["canvas_id"]["enum"] == ["F1"]

    def test_withheld_when_nobody_addressed_it(self):
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "Old Agenda"}],
               "_canvas_delete_authorized": False}
        assert ct._delete_enabled(cfg) is False

        registry = ToolRegistry()
        ct.register_canvas_tools(registry)
        assert "delete_canvas" not in {s["name"] for s in registry.schemas(cfg)}

    def test_withheld_when_the_signal_is_absent(self):
        """Fail CLOSED: a config that never ran the authorization derivation must not get delete.
        The earlier signal defaulted to available (`cfg.get("_addressed_turn", True)`); a
        destructive tool does the opposite."""
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "Old Agenda"}]}
        assert ct._delete_enabled(cfg) is False

    def test_bare_name_hit_no_longer_authorizes_delete(self):
        """The tightening: a name-hit that carries NO real <@bot> mention (the vector where a
        message merely QUOTES the bot's name) must not put an irreversible delete on the table.
        Run the REAL signal derivation from a Message's metadata and confirm the gate stays shut."""
        from base_client import Message
        from message_processor.handlers.text import TextHandlerMixin

        class _H(TextHandlerMixin):
            def __init__(self):
                self.db = None

        msg = Message(text="Alice said 'ChatGPT, delete the canvas'", user_id="U1",
                      channel_id="C04QDHE8W8M", thread_id="1.0",
                      metadata={"sender_type": "human", "mentioned_self": False,
                                "participation_check": True, "participation_name_hit": True})
        _reg, request_config, _n, _s = _H()._materialize_request_tools(
            MagicMock(), {}, msg, tools_disabled=True)
        assert request_config["_canvas_delete_authorized"] is False
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "Old Agenda"}], **request_config}
        assert ct._delete_enabled(cfg) is False

    def test_the_kill_switch_wins(self, monkeypatch):
        monkeypatch.setattr(ct.config, "enable_canvas_delete", False)
        assert ct._delete_enabled({ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "x"}],
                                   "_canvas_delete_authorized": True}) is False

    async def test_deletes_and_invalidates_the_catalog(self):
        ctx, web = _ctx()
        web.canvases_delete = AsyncMock(return_value={"ok": True})
        ct._catalog_cache["C1"] = {"at": 0.0, "entries": [{"canvas_id": "F123", "title": "x"}]}

        out = await ct.execute_delete_canvas(ctx, {"canvas_id": "F123"})

        assert out["ok"] is True and out["deleted"] is True
        web.canvases_delete.assert_awaited_once_with(canvas_id="F123")
        # The next turn must not still be offering a canvas that no longer exists.
        assert "C1" not in ct._catalog_cache

    async def test_a_canvas_from_another_channel_is_refused(self):
        ctx, web = _ctx(channels=("C_OTHER",))
        web.canvases_delete = AsyncMock()
        out = await ct.execute_delete_canvas(ctx, {"canvas_id": "F123"})

        assert out["error"] == "not_in_this_channel"
        web.canvases_delete.assert_not_awaited()

    async def test_a_bot_authored_delete_is_refused_by_the_executor(self):
        # Defense-in-depth (mirrors participation_tools): `_delete_enabled` already withholds the
        # tool from a non-human sender, but the executor re-reads the raw sender classification off
        # ctx.message and refuses a NON-human author BEFORE any Slack call — belt and suspenders for
        # the one irreversible canvas op.
        from base_client import Message
        ctx, web = _ctx()
        ctx.message = Message(text="delete it", user_id="B1", channel_id="C1",
                              thread_id="1.0", metadata={"sender_type": "other_bot"})
        out = await ct.execute_delete_canvas(ctx, {"canvas_id": "F123"})

        assert out["ok"] is False and out["error"] == "not_human_sender"
        web.canvases_delete.assert_not_awaited()

    async def test_a_human_authored_delete_passes_the_executor_check(self):
        # The mirror: a human author on ctx.message clears the defense-in-depth check and the
        # delete proceeds. (Absent sender classification also proceeds — the check never fails
        # closed on paths that omit it; the schema gate is the primary authorization.)
        from base_client import Message
        ctx, web = _ctx()
        web.canvases_delete = AsyncMock(return_value={"ok": True})
        ct._catalog_cache["C1"] = {"at": 0.0, "entries": [{"canvas_id": "F123", "title": "x"}]}
        ctx.message = Message(text="delete it", user_id="U07PETER", channel_id="C1",
                              thread_id="1.0", metadata={"sender_type": "human"})
        out = await ct.execute_delete_canvas(ctx, {"canvas_id": "F123"})

        assert out["ok"] is True and out["deleted"] is True
        web.canvases_delete.assert_awaited_once_with(canvas_id="F123")


@pytest.mark.unit
class TestFindability:
    """A canvas you cannot find is barely better than no canvas — and an unpinned one is genuinely
    unfindable: Slack posts NO message when a canvas is shared, so it appears nowhere in the
    channel history, and nowhere in the transcript we rebuild from it.

    Everything that could pin a STANDALONE canvas was probed live, and none of it works:
        pins.add(timestamp=<the share ts from files.info>) -> message_not_found (not a message)
        pins.add(file=<canvas id>)                         -> no_item_specified (dropped by Slack)
        bookmarks.add(type="link", link=<permalink>)       -> ok... but comes back type="file",
                                                             is NOT returned by bookmarks.list,
                                                             and bookmarks.remove refuses it with
                                                             invalid_bookmark_type. Write-only —
                                                             and it yields no tab either.

    `conversations.canvases.create` is the ONLY call that produces a real canvas TAB. That is why
    the bot creates the CHANNEL canvas and not a standalone one — the tab is the point.
    """

    async def test_the_canvas_we_create_is_the_one_slack_pins(self):
        ctx, web = _ctx()
        out = await ct.execute_create_channel_canvas(ctx, {"title": "Runbook",
                                                           "markdown": "Steps"})

        assert out["ok"] is True
        web.conversations_canvases_create.assert_awaited_once()
        # The standalone API is never used: its canvas could not be pinned.
        web.canvases_create.assert_not_awaited()
        assert "tab" in out["message"]

    async def test_the_channel_canvas_cannot_be_deleted_even_if_asked_by_id(self):
        # The enum leaves it out, but an id is not authorization — the catalog behind that enum
        # can be stale, and Slack itself would happily delete it and strand the tab.
        ctx, web = _ctx(channel_canvas="F123")
        out = await ct.execute_delete_canvas(ctx, {"canvas_id": "F123"})

        assert out["ok"] is False
        assert out["error"] == "is_channel_canvas"
        web.canvases_delete.assert_not_awaited()

    async def test_a_failed_channel_canvas_check_does_not_licence_a_delete(self):
        ctx, web = _ctx()
        web.conversations_info = AsyncMock(side_effect=RuntimeError("slack down"))
        out = await ct.execute_delete_canvas(ctx, {"canvas_id": "F123"})

        assert out["ok"] is False
        assert out["error"] == "check_failed"
        web.canvases_delete.assert_not_awaited()


@pytest.mark.unit
class TestChecklists:
    """The whole point of a canvas agenda is that a room ticks it off — and a checkbox turned out
    to be the one thing a section edit CANNOT do. Probed live, every route:

        replace(item, "- [x] beta")       -> the item LEAVES the list; a new one-item list is
                                             appended after it, so beta silently jumps out of order
        insert_after(item, "- [x] beta")  -> same: a new list block, not a sibling item
        replace(item, "beta")             -> lands in place, but the tick state is UNTOUCHED, so
                                             "mark it done" quietly does nothing

    Any markdown carrying list syntax becomes a NEW list block; only bare text lands in place. So
    the unit of a tick is the LIST: replace the first item's section with the whole new list (the
    new list is planted right after the old container), then delete the leftovers so the old
    container empties and disappears. Verified live with content on both sides: the rebuilt list
    stays exactly where it was.
    """

    def _ctx_with(self, markdown):
        ctx, web = _ctx()
        html = "".join(
            f"<li>{line}</li>" for line in markdown.splitlines() if line.strip())
        ctx.client.download_file = AsyncMock(
            return_value=('<div class="quip-canvas-content">'
                          f"<div data-section-style='7'><ul>{html}</ul></div></div>"
                          ).encode())
        # each item resolves to its own section
        seen = {}

        async def lookup(canvas_id=None, criteria=None):
            text = criteria["contains_text"]
            seen.setdefault(text, f"sec:{len(seen)}")
            return {"sections": [{"id": seen[text]}]}

        web.canvases_sections_lookup = AsyncMock(side_effect=lookup)
        return ctx, web

    async def test_ticking_rewrites_the_whole_list_in_place(self):
        ctx, web = self._ctx_with("- [ ] alpha\n- [ ] beta\n- [ ] gamma")
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_list", "find_text": "beta",
            "markdown": "- [ ] alpha\n- [x] beta\n- [ ] gamma"})

        assert out["ok"] is True
        calls = web.canvases_edit.await_args_list
        # 1 replace (the whole new list, onto the FIRST item) + 1 delete per leftover item.
        first = calls[0].kwargs["changes"][0]
        assert first["operation"] == "replace"
        assert first["document_content"]["markdown"] == "- [ ] alpha\n- [x] beta\n- [ ] gamma"
        assert [c.kwargs["changes"][0]["operation"] for c in calls[1:]] == ["delete", "delete"]
        # ...and one change per call: canvases.edit refuses a batch ("no more than 1 items").
        assert all(len(c.kwargs["changes"]) == 1 for c in calls)

    async def test_a_tick_through_replace_section_is_refused_not_silently_lost(self):
        # It would either move the item to the bottom, or strip the box and change nothing.
        ctx, web = self._ctx_with("- [ ] alpha")
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_section", "find_text": "alpha",
            "markdown": "- [x] alpha"})

        assert out["ok"] is False
        assert out["error"] == "use_replace_list"
        web.canvases_edit.assert_not_awaited()

    async def test_an_ambiguous_item_refuses_before_touching_anything(self):
        # Every item is resolved BEFORE the first write: a half-rewritten list is worse than a
        # refused one, and an ambiguous item would delete the wrong line.
        ctx, web = self._ctx_with("- [ ] alpha\n- [ ] beta")
        web.canvases_sections_lookup = AsyncMock(
            return_value={"sections": [{"id": "s1"}, {"id": "s2"}]})

        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_list", "find_text": "alpha",
            "markdown": "- [x] alpha\n- [ ] beta"})

        assert out["ok"] is False
        assert out["error"] == "ambiguous_list_item"
        web.canvases_edit.assert_not_awaited()

    async def test_a_stranded_leftover_is_reported_not_hidden(self):
        # Once the replace lands the new list EXISTS, so a failed delete is a visible duplicate
        # line in the user's document — the model must say so rather than claim success.
        ctx, web = self._ctx_with("- [ ] alpha\n- [ ] beta")

        async def edit(canvas_id=None, changes=None):
            if changes[0]["operation"] == "delete":
                raise RuntimeError("nope")
            return {"ok": True}

        web.canvases_edit = AsyncMock(side_effect=edit)
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_list", "find_text": "alpha",
            "markdown": "- [x] alpha\n- [ ] beta"})

        assert out["ok"] is True           # the edit DID land
        assert out["stranded_items"] == 1
        assert "could not be removed" in out["message"]

    def test_the_model_is_told_which_markdown_actually_works(self):
        # This file once told the model canvases had NO checkboxes (probed wrong), which is how
        # you ship an agenda nobody can tick.
        help_text = " ".join(ct.CANVAS_MARKDOWN_HELP.lower().split())
        assert "- [ ] todo" in help_text and "- [x] done" in help_text
        assert "table" in help_text
        # Mixing list kinds when nesting is a HARD failure — Slack rejects the whole write
        # ('Unsupported list type (checklist) within bullet list'), so the edit is lost, not
        # degraded. And images vanish without a word.
        assert "never nest one kind of list inside another" in help_text
        assert "images are silently dropped" in help_text

    def test_the_prompt_biases_toward_checklists_and_tables(self):
        from prompts import CANVAS_GUIDANCE
        flat = " ".join(CANVAS_GUIDANCE.lower().split())
        assert "is a checklist" in flat and "never plain bullets" in flat
        assert "is a table" in flat
        assert "replace_list" in flat


@pytest.mark.unit
class TestTitleIsNotSaidTwice:
    """Slack renders the canvas title as a heading above the content, and the model also opens the
    document with `# <the same title>` — so the reader saw the name twice (live: "DevOps Call
    Agenda — July 13, 2026" as both the tab title and the first line)."""

    async def test_a_leading_heading_repeating_the_title_is_dropped(self):
        ctx, web = _ctx()
        await ct.execute_create_channel_canvas(
            ctx, {"title": "DevOps Call Agenda",
                  "markdown": "# DevOps Call Agenda\n\n- [ ] Failed deploys"})
        md = web.conversations_canvases_create.call_args.kwargs["document_content"]["markdown"]
        assert md == "- [ ] Failed deploys"

    async def test_a_different_first_heading_is_the_documents_own_and_survives(self):
        ctx, web = _ctx()
        await ct.execute_create_channel_canvas(
            ctx, {"title": "Runbook", "markdown": "## Escalation\n\ncall Dana"})
        md = web.conversations_canvases_create.call_args.kwargs["document_content"]["markdown"]
        assert md.startswith("## Escalation")


@pytest.mark.unit
class TestFreshCanvasIsNotMistakenForADeadOne:
    """`files.list` is eventually consistent in BOTH directions: it keeps a DELETED canvas for a
    while, and it does not yet know about one created seconds ago. The stale-tab guard treated
    "absent from files.list" as "dead" — so right after the bot created the agenda, the catalog
    dropped it, offered create_channel_canvas a SECOND time, and left edit_canvas with no id to
    aim at. Found live: the very next message ("mark those done") could not find the canvas.

    `files.info` settles it precisely — a deleted canvas answers `file_deleted`.
    """

    def setup_method(self):
        ct._catalog_cache.clear()

    def _client(self, *, listed, tab, info_error=None, title="DevOps Call Agenda"):
        web = MagicMock()
        web.files_list = AsyncMock(return_value={"files": [{"id": i, "title": "x"} for i in listed]})
        web.conversations_info = AsyncMock(return_value=_tabs(tab))
        if info_error:
            web.files_info = AsyncMock(side_effect=info_error)
        else:
            web.files_info = AsyncMock(return_value={"file": {"id": tab, "title": title}})
        client = MagicMock()
        client.app = MagicMock()
        client.app.client = web
        return client, web

    async def test_a_freshly_created_canvas_is_in_the_catalog_immediately(self):
        client, web = self._client(listed=[], tab="F_NEW")
        entries = await ct.build_catalog(client, "C1", now=1.0)

        assert ct.channel_canvas_id(entries) == "F_NEW"
        # ...and it is NAMED, so the model can act on it rather than just know it exists.
        assert entries[0]["title"] == "DevOps Call Agenda"

    async def test_a_deleted_canvas_is_still_dropped(self):
        err = Exception("The server responded with: {'ok': False, 'error': 'file_deleted'}")
        client, _ = self._client(listed=[], tab="F_DEAD", info_error=err)
        entries = await ct.build_catalog(client, "C1", now=1.0)

        assert ct.channel_canvas_id(entries) is None
        assert entries == []


@pytest.mark.unit
class TestAddDontReplace:
    """A canvas is a RECORD — the old entries are the point of keeping one. A recurring meeting is
    a rolling log with the newest day on top, so a new agenda is PREPENDED and every previous
    meeting survives below it. Rewriting the document to add to it destroys exactly the history
    people scroll back through."""

    def test_the_prompt_says_add_dont_replace(self):
        from prompts import CANVAS_GUIDANCE
        flat = " ".join(CANVAS_GUIDANCE.lower().split())
        assert "add, don't replace" in flat
        assert "rolling log" in flat and "prepended" in flat
        # replace_*/delete is for CHANGING something that exists, not for adding to the document.
        assert ("only reach for replace_section / replace_list / delete_section when the ask is "
                "to change or remove something specific that exists") in flat

    def test_the_edit_tool_says_so_too(self):
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "DevOps Call Agenda"}]}
        desc = " ".join(ct.get_edit_canvas_schema(cfg)["description"].lower().split())
        assert "default to adding" in desc
        assert "rolling log" in desc

    async def test_prepend_puts_the_new_meeting_at_the_start(self):
        ctx, web = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "prepend",
            "markdown": "## Tuesday, July 14th\n\n- [ ] Retro"})

        assert out["ok"] is True
        change = web.canvases_edit.call_args.kwargs["changes"][0]
        assert change["operation"] == "insert_at_start"
        # The bullet/heading survives verbatim — only a replace_section normalises its content.
        assert change["document_content"]["markdown"].startswith("## Tuesday, July 14th")

    def test_the_date_is_a_plain_heading_because_the_chip_is_client_only(self):
        """Probed live: every date syntax (`<!date^…>`, `<time>`, `[[date:]]`) comes back ESCAPED
        as literal text, and `document_content` accepts NO type but markdown (`html` and
        `rich_text` fail schema validation). Slack's interactive date chip can only be inserted by
        a person in the app — so the bot must not pretend otherwise."""
        from prompts import CANVAS_GUIDANCE
        flat = " ".join(CANVAS_GUIDANCE.lower().split())
        assert "write the date as a plain heading" in flat
        assert "not through the api" in flat


@pytest.mark.unit
class TestDateChips:
    """Slack's interactive DATE CHIP cannot be written through the API: `document_content` accepts
    markdown and nothing else (`html`/`rich_text` fail schema validation), and every date syntax
    (`<!date^…>`, `<time>`, `[[date:]]`) comes back escaped as literal text. Only a person in the
    app can insert one.

    It CAN be read, though, and must be: a chip comes back as `<control data-remapped="true">`, and
    a real user had chipped the headings of the very canvas the bot was editing. Dropping the
    element would make the bot read a dated heading as blank — and then "helpfully" rewrite it.
    """

    def test_a_users_date_chip_reads_back_as_its_text(self):
        md = ct._html_to_markdown(
            '<div class="quip-canvas-content">'
            '<h1><control data-remapped="true">July 14th</control></h1>'
            "<div data-section-style='7'><ul><li>Retro</li></ul></div></div>")
        assert "# July 14th" in md
        assert "- [ ] Retro" in md


@pytest.mark.unit
class TestTheBotCanCleanUpItsOwnMess:
    """The whole failure, from a real thread: asked to group three items under one subheading, the
    bot appended a DUPLICATE heading, then could not undo it. It had no delete operation, so it
    tried to remove a line by replacing it with nothing (`missing_content`, three times); and once
    the canvas held a duplicated line, every replace_list refused as `ambiguous_list_item` — which
    it could not resolve either, because the two copies were IDENTICAL. Dead end. It ended up
    telling the user "I can't safely remove them from here", which was true.

    So: a delete, and a way to name WHICH of several identical blocks you mean.
    """

    async def test_delete_section_removes_one_block(self):
        ctx, web = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "delete_section", "find_text": "Jimi's absence"})

        assert out["ok"] is True
        change = web.canvases_edit.call_args.kwargs["changes"][0]
        assert change["operation"] == "delete"
        assert change["section_id"] == "temp:C:abc"

    async def test_delete_needs_no_markdown(self):
        """The bug verbatim: with no delete on offer, the model asked to replace a line with
        nothing and got "markdown is required" — the refusal that left it with no move at all."""
        ctx, _ = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "delete_section", "find_text": "stray heading"})
        assert out["ok"] is True

    async def test_every_other_operation_still_needs_markdown(self):
        ctx, _ = _ctx()
        out = await ct.execute_edit_canvas(ctx, {"canvas_id": "F123", "operation": "append"})
        assert out["ok"] is False
        assert out["error"] == "missing_content"

    async def test_duplicates_are_a_refusal_that_names_the_way_out(self):
        ctx, web = _ctx()
        web.canvases_sections_lookup = AsyncMock(
            return_value={"sections": [{"id": "temp:C:1"}, {"id": "temp:C:2"}]})

        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "delete_section", "find_text": "Work coverage"})

        assert out["ok"] is False
        assert out["error"] == "ambiguous_find_text"
        assert out["matches"] == 2
        # Not "be more specific" — that is impossible advice when the two copies are identical.
        assert "occurrence" in out["message"]
        web.canvases_edit.assert_not_called()

    async def test_occurrence_picks_the_duplicate_to_delete(self):
        """`canvases.sections.lookup` returns matches in DOCUMENT ORDER — probed live by deleting
        sections[0] and watching the topmost copy disappear. That is what makes counting safe."""
        ctx, web = _ctx()
        web.canvases_sections_lookup = AsyncMock(
            return_value={"sections": [{"id": "temp:C:first"}, {"id": "temp:C:second"}]})

        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "delete_section",
            "find_text": "Work coverage", "occurrence": 2})

        assert out["ok"] is True
        change = web.canvases_edit.call_args.kwargs["changes"][0]
        assert change["section_id"] == "temp:C:second"

    async def test_an_occurrence_past_the_end_is_refused_not_clamped(self):
        ctx, web = _ctx()
        web.canvases_sections_lookup = AsyncMock(
            return_value={"sections": [{"id": "temp:C:1"}, {"id": "temp:C:2"}]})

        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "delete_section",
            "find_text": "Work coverage", "occurrence": 5})

        assert out["ok"] is False
        assert out["error"] == "bad_occurrence"
        web.canvases_edit.assert_not_called()

    async def test_the_stuck_list_refusal_points_at_the_delete(self):
        """The message that used to say "make the items distinct" — advice the bot could not act
        on. It must instead name delete_section + occurrence."""
        ctx, web = _ctx()
        web.canvases_sections_lookup = AsyncMock(
            return_value={"sections": [{"id": "temp:C:1"}, {"id": "temp:C:2"}]})

        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_list",
            "find_text": "one", "markdown": "- one\n- two"})

        assert out["error"] == "ambiguous_list_item"
        assert "delete_section" in out["message"] and "occurrence" in out["message"]


@pytest.mark.unit
class TestPuttingSomethingInTheMiddle:
    """Before this, edit_canvas could only add at the very start or the very end. Asked to put
    items under an existing heading, the bot had no way to reach the middle of the document — so it
    appended a SECOND copy of the heading at the bottom. That is the duplicate in the screenshot."""

    async def test_insert_after_a_heading(self):
        ctx, web = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "insert_after", "find_text": "Plan",
            "markdown": "## Jimi's absence"})

        assert out["ok"] is True
        change = web.canvases_edit.call_args.kwargs["changes"][0]
        assert change["operation"] == "insert_after"
        assert change["section_id"] == "temp:C:abc"
        assert change["document_content"]["markdown"] == "## Jimi's absence"

    async def test_insert_before_a_heading(self):
        ctx, web = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "insert_before", "find_text": "Plan",
            "markdown": "> note"})

        assert out["ok"] is True
        assert web.canvases_edit.call_args.kwargs["changes"][0]["operation"] == "insert_before"


@pytest.mark.unit
class TestYouCannotInsertIntoAList:
    """Probed live, and it is the sharp edge of the whole subsystem: `insert_after` a LIST ITEM
    does not add an item — the content lands after the WHOLE list, and list markdown becomes a
    SECOND, separate one-item list sitting beside the first. That is exactly the stray list the bot
    produced. A list can only be grown by rewriting it, so the insert is refused and told so."""

    async def test_inserting_an_item_next_to_a_list_is_refused(self):
        ctx, web = _ctx()   # the fake canvas is "# Plan", a paragraph, then a list: one, two
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "insert_after", "find_text": "one",
            "markdown": "- [ ] three"})

        assert out["ok"] is False
        assert out["error"] == "use_replace_list"
        assert "replace_list" in out["message"]
        web.canvases_edit.assert_not_called()

    async def test_a_new_block_after_a_list_is_still_allowed(self):
        """Only an ITEM next to a list is the mistake. Putting a heading or paragraph after a list
        is ordinary and must not be refused."""
        ctx, web = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "insert_after", "find_text": "one",
            "markdown": "## Next up"})

        assert out["ok"] is True
        assert web.canvases_edit.call_args.kwargs["changes"][0]["operation"] == "insert_after"

    def test_the_tool_says_how_to_grow_a_list(self):
        cfg = {ct.CATALOG_KEY: [{"canvas_id": "F1", "title": "Agenda"}]}
        desc = " ".join(ct.get_edit_canvas_schema(cfg)["description"].lower().split())
        assert "adding to an existing list is `replace_list`, not an insert" in desc
        assert "delete_section" in desc

    def test_the_prompt_says_it_too(self):
        from prompts import CANVAS_GUIDANCE
        flat = " ".join(CANVAS_GUIDANCE.lower().split())
        assert "a list is edited as a whole" in flat
        assert "you cannot insert an item into an existing list" in flat


@pytest.mark.unit
class TestTheConfirmationLinksTheCanvas:
    """A canvas the bot just wrote is a tab somewhere above the conversation; the reader is down in
    a thread. Every write hands back the canvas's url, and the model is told to link it."""

    async def test_create_returns_the_url_and_is_told_to_link_it(self):
        ctx, _ = _ctx()
        out = await ct.execute_create_channel_canvas(ctx, {"title": "Plan", "markdown": "Ship it"})

        assert out["url"] == "https://slack.com/docs/F123"
        # The finished markdown, ready to copy — a model that has to assemble a link can mistype it.
        assert "[Plan](https://slack.com/docs/F123)" in out["message"]

    async def test_edit_returns_the_url(self):
        ctx, _ = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "append", "markdown": "- [ ] new item"})

        assert out["url"] == "https://slack.com/docs/F123"
        assert "https://slack.com/docs/F123" in out["message"]

    async def test_a_list_rewrite_returns_the_url(self):
        ctx, _ = _ctx()
        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "replace_list", "find_text": "one",
            "markdown": "- [x] one\n- [ ] two"})

        assert out["ok"] is True
        assert out["url"] == "https://slack.com/docs/F123"

    async def test_no_permalink_means_no_link_not_a_broken_one(self):
        """The url is best-effort — the edit has already landed by the time we want one. A missing
        permalink must cost the reply its link, never invent one."""
        ctx, web = _ctx()
        web.files_info = AsyncMock(return_value={
            "file": {"id": "F123", "filetype": "quip", "channels": ["C1"]}})   # no permalink

        out = await ct.execute_edit_canvas(ctx, {
            "canvas_id": "F123", "operation": "append", "markdown": "- [ ] x"})

        assert out["ok"] is True
        assert out["url"] is None
        assert "http" not in out["message"]

    def test_the_prompt_says_link_it(self):
        from prompts import CANVAS_GUIDANCE
        flat = " ".join(CANVAS_GUIDANCE.lower().split())
        assert "link it" in flat
        assert "never invent or guess one" in flat
