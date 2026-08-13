"""search_stored_knowledge — asking the derived stores what this channel already worked out.

The properties under test are the ones that decide whether a hit can be trusted and acted on:
the canonical authorization gate runs before any stored text leaves, channel scope comes from
the ``channel:thread`` key prefix (colons and all), document handles are the ones read_document
genuinely resolves while image hits carry no viewing id at all, an empty result says out loud
that it is not proof of absence, and a store that failed is named rather than folded into a
confident zero.
"""
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor import knowledge_tools
from tool_registry import ToolContext, ToolRegistry

CHANNEL = "C0KNOW123"
DM = "D0KNOW456"


def _doc(**kw: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": 1,
        "thread_id": f"{CHANNEL}:1740787200.000100",
        "filename": "q1-pricing.xlsx",
        "mime_type": "application/vnd.ms-excel",
        "summary": "Quarterly pricing sheet; the wholesale column is blank for March.",
        "file_id": "F0DOC0001",
        "message_ts": "1740787200.000100",
        "created_at": "2026-03-02 10:00:00",
    }
    row.update(kw)
    return row


def _img(**kw: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": 7,
        "thread_id": f"{CHANNEL}:1740790000.000200",
        "url": "https://files.slack.test/shot.png",
        "image_type": "screenshot",
        "prompt": "",
        "analysis": "A browser devtools panel showing a 500 error on the checkout endpoint.",
        "original_analysis": None,
        "message_ts": "1740790000.000200",
        "created_at": "2026-03-04 09:00:00",
    }
    row.update(kw)
    return row


class _DB:
    """A database double recording the exact arguments each accessor was called with."""

    def __init__(self, docs: Optional[List[Dict]] = None, images: Optional[List[Dict]] = None,
                 doc_error: Optional[Exception] = None,
                 image_error: Optional[Exception] = None):
        self._docs = list(docs or [])
        self._images = list(images or [])
        self._doc_error = doc_error
        self._image_error = image_error
        self.doc_calls: List[tuple] = []
        self.image_calls: List[tuple] = []

    async def search_channel_documents_async(self, channel_id, query, limit=10):
        self.doc_calls.append((channel_id, query, limit))
        if self._doc_error:
            raise self._doc_error
        return list(self._docs)

    async def search_channel_image_analyses_async(self, channel_id, query, limit=10):
        self.image_calls.append((channel_id, query, limit))
        if self._image_error:
            raise self._image_error
        return list(self._images)


def _client(verdict: str = "ALLOW", permalink: Optional[str] = "https://slack.test/archives/p1"):
    web = MagicMock()
    if permalink is None:
        web.chat_getPermalink = AsyncMock(side_effect=RuntimeError("no link for you"))
    else:
        web.chat_getPermalink = AsyncMock(return_value={"ok": True, "permalink": permalink})
    client = MagicMock()
    client.app = SimpleNamespace(client=web)
    client._authorize_channel_read = AsyncMock(return_value=(verdict, "both_members"))
    return client


def _ctx(client, db, channel_id: str = CHANNEL, is_dm: bool = False) -> ToolContext:
    return ToolContext(channel_id=channel_id, thread_ts="1740787200.000100", client=client,
                       db=db, user_id="U0ASKER01", requester_is_human=True, is_dm=is_dm)


async def _run(client, db, args: Optional[Dict[str, Any]] = None,
               channel_id: str = CHANNEL) -> Dict[str, Any]:
    ctx = _ctx(client, db, channel_id=channel_id)
    return await knowledge_tools.execute_search_stored_knowledge(
        ctx, args if args is not None else {"query": "pricing"})


@pytest.mark.unit
@pytest.mark.asyncio
class TestAuthorization:
    async def test_a_denied_read_returns_the_canonical_refusal_and_queries_nothing(self):
        from slack_client.history_tool import ACCESS_DENIED_MESSAGE

        db = _DB(docs=[_doc()])
        result = await _run(_client(verdict="DENY"), db)

        assert result["ok"] is False and result["error"] == "not_accessible"
        assert result["message"] == ACCESS_DENIED_MESSAGE
        # The gate is not advisory: nothing was read, so nothing could leak through the result.
        assert db.doc_calls == [] and db.image_calls == []

    async def test_a_redirect_is_indistinguishable_from_a_denial(self):
        deny = await _run(_client(verdict="DENY"), _DB(docs=[_doc()]))
        redirect = await _run(_client(verdict="REDIRECT"), _DB(docs=[_doc()]))
        assert deny == redirect

    async def test_the_gate_is_asked_about_this_conversation(self):
        client = _client(verdict="DENY")
        await _run(client, _DB(), channel_id="C0OTHER99")
        assert client._authorize_channel_read.await_args.args[0] == "C0OTHER99"

    async def test_a_client_without_the_gate_refuses_rather_than_skipping_it(self):
        client = _client()
        del client._authorize_channel_read
        client.mock_add_spec(["app"])
        db = _DB(docs=[_doc()])
        result = await knowledge_tools.execute_search_stored_knowledge(
            _ctx(client, db), {"query": "pricing"})
        assert result["ok"] is False and result["error"] == "unavailable"
        assert db.doc_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestScopeAndArguments:
    async def test_both_stores_are_searched_for_this_channel(self):
        db = _DB(docs=[_doc()], images=[_img()])
        await _run(_client(), db, {"query": "500 error"})
        assert db.doc_calls == [(CHANNEL, "500 error", 10)]
        assert db.image_calls == [(CHANNEL, "500 error", 10)]

    async def test_a_dm_scopes_to_that_dm(self):
        db = _DB()
        await _run(_client(), db, {"query": "x"}, channel_id=DM)
        assert db.doc_calls[0][0] == DM and db.image_calls[0][0] == DM

    @pytest.mark.parametrize("given,expected", [
        (None, 10), (5, 5), (0, 1), (99, 20), ("7", 7), ("nonsense", 10),
    ])
    async def test_the_limit_follows_the_search_tools_convention(self, given, expected):
        db = _DB()
        args: Dict[str, Any] = {"query": "x"}
        if given is not None:
            args["limit"] = given
        await _run(_client(), db, args)
        assert db.doc_calls[0][2] == expected

    async def test_an_empty_query_is_refused_without_a_lookup(self):
        db = _DB(docs=[_doc()])
        result = await _run(_client(), db, {"query": "   "})
        assert result["ok"] is False and result["error"] == "missing_query"
        assert db.doc_calls == []

    async def test_the_query_is_stripped_before_it_reaches_the_store(self):
        db = _DB()
        await _run(_client(), db, {"query": "  pricing  "})
        assert db.doc_calls[0][1] == "pricing"


@pytest.mark.unit
@pytest.mark.asyncio
class TestResultContract:
    async def test_a_document_hit_carries_the_handle_read_document_resolves(self):
        result = await _run(_client(), _DB(docs=[_doc()]), {"query": "pricing"})
        hit = result["results"][0]
        assert hit["kind"] == "document"
        # read_document takes a file_id or a filename — both are real, resolvable inputs.
        assert hit["file_id"] == "F0DOC0001"
        assert hit["filename"] == "q1-pricing.xlsx"
        assert "read_document" in result["how_to_use"]

    async def test_the_thread_ts_survives_the_colon_in_the_thread_key(self):
        result = await _run(_client(), _DB(docs=[_doc()]), {"query": "pricing"})
        # The key is "channel:thread"; the thread half must come back whole and unprefixed.
        assert result["results"][0]["thread_ts"] == "1740787200.000100"

    async def test_a_row_from_another_channels_key_gets_no_thread_ts(self):
        stray = _doc(thread_id="C0ELSEWHERE:1740787200.000100")
        result = await _run(_client(), _DB(docs=[stray]), {"query": "pricing"})
        # Mislabelling it as this channel's thread would be worse than omitting the field.
        assert "thread_ts" not in result["results"][0]

    async def test_an_image_hit_is_informational_with_no_viewing_id(self):
        result = await _run(_client(), _DB(images=[_img()]), {"query": "500 error"})
        hit = result["results"][0]
        assert hit["kind"] == "image"
        assert "500 error" in hit["analysis_snippet"]
        assert hit["permalink"] == "https://slack.test/archives/p1"
        # view_image resolves only this turn's catalog ids, so a synthesised one would always
        # break. No id field may appear on an image hit at all.
        assert not any(key in hit for key in ("image_id", "id", "file_id"))
        assert "no id" in result["how_to_use"]

    async def test_the_snippet_comes_from_the_column_that_actually_matched(self):
        # The SQL matches analysis OR original_analysis. When the PRE-EDIT text is what matched,
        # quoting the current analysis would hand back evidence with no trace of the search term.
        edited = _img(analysis="Now with a blue roof and a new sign.",
                      original_analysis="A red barn standing in deep snow.")
        result = await _run(_client(), _DB(images=[edited]), {"query": "red barn"})
        hit = result["results"][0]
        assert "red barn" in hit["analysis_snippet"].lower()
        assert "blue roof" not in hit["analysis_snippet"]
        assert hit["matched_in"] == "pre-edit description"

    async def test_the_current_description_is_preferred_when_it_matches(self):
        edited = _img(analysis="A blue roof over the barn.",
                      original_analysis="A blue roof in an older render.")
        result = await _run(_client(), _DB(images=[edited]), {"query": "blue roof"})
        hit = result["results"][0]
        assert hit["matched_in"] == "description"
        assert "over the barn" in hit["analysis_snippet"]

    async def test_a_missing_permalink_leaves_the_hit_without_a_broken_link(self):
        result = await _run(_client(permalink=None), _DB(images=[_img()]), {"query": "500 error"})
        hit = result["results"][0]
        assert "permalink" not in hit
        assert result["ok"] is True and hit["analysis_snippet"]

    async def test_matched_in_distinguishes_a_summary_hit_from_a_filename_hit(self):
        by_summary = await _run(_client(), _DB(docs=[_doc()]), {"query": "wholesale"})
        by_name = await _run(_client(), _DB(docs=[_doc()]), {"query": "xlsx"})
        assert by_summary["results"][0]["matched_in"] == "summary"
        assert by_name["results"][0]["matched_in"] == "filename"

    async def test_a_filename_only_match_says_there_is_no_summary_rather_than_quoting_one(self):
        row = _doc(summary=None)
        result = await _run(_client(), _DB(docs=[row]), {"query": "pricing"})
        hit = result["results"][0]
        assert "summary_snippet" not in hit
        assert "no summary" in hit["note"]

    async def test_a_filename_only_match_never_quotes_an_unrelated_summary(self):
        # The filename carries the term and the summary is about something else entirely; the
        # head of that summary shipped as `summary_snippet` reads as evidence for this match.
        row = _doc(filename="pricing.xlsx", summary="Employee holiday calendar for the year.")
        result = await _run(_client(), _DB(docs=[row]), {"query": "pricing"})
        hit = result["results"][0]
        assert hit["matched_in"] == "filename"
        assert "summary_snippet" not in hit
        assert "holiday calendar" not in str(hit)
        assert "filename" in hit["note"]

    async def test_results_from_both_stores_are_merged_newest_first(self):
        old_doc = _doc(created_at="2026-03-01 10:00:00")
        new_image = _img(created_at="2026-03-09 10:00:00")
        result = await _run(_client(), _DB(docs=[old_doc], images=[new_image]), {"query": "e"})
        assert [h["kind"] for h in result["results"]] == ["image", "document"]

    async def test_the_merged_list_honours_the_requested_limit(self):
        docs = [_doc(id=i, file_id=f"F{i}", created_at=f"2026-03-0{i} 10:00:00")
                for i in range(1, 4)]
        images = [_img(id=i, created_at=f"2026-03-0{i} 11:00:00") for i in range(1, 4)]
        result = await _run(_client(), _DB(docs=docs, images=images),
                            {"query": "e", "limit": 2})
        assert result["count"] == 2 and len(result["results"]) == 2


@pytest.mark.unit
@pytest.mark.asyncio
class TestHonestyAboutMisses:
    async def test_an_empty_result_says_it_is_not_proof_the_document_lacks_it(self):
        result = await _run(_client(), _DB(), {"query": "nothing here"})
        assert result["ok"] is True and result["count"] == 0
        # The whole point of the caveat: summaries are searched, contents are not.
        assert "does not prove" in result["note"]
        assert "read_document" in result["note"]

    async def test_one_failed_store_is_named_instead_of_reported_as_zero(self):
        db = _DB(docs=[_doc()], image_error=RuntimeError("db locked"))
        result = await _run(_client(), db, {"query": "pricing"})
        assert result["ok"] is True
        assert len(result["results"]) == 1
        assert "image descriptions" in result["incomplete"]

    async def test_both_stores_failing_is_a_failure_not_an_empty_answer(self):
        db = _DB(doc_error=RuntimeError("boom"), image_error=RuntimeError("boom"))
        result = await _run(_client(), db, {"query": "pricing"})
        assert result["ok"] is False and result["error"] == "lookup_failed"

    async def test_the_executor_never_raises(self):
        client = _client()
        client._authorize_channel_read = AsyncMock(side_effect=RuntimeError("gate exploded"))
        result = await knowledge_tools.execute_search_stored_knowledge(
            _ctx(client, _DB()), {"query": "pricing"})
        assert result["ok"] is False and result["error"] == "search_failed"

    async def test_a_context_without_a_database_refuses_cleanly(self):
        result = await knowledge_tools.execute_search_stored_knowledge(
            _ctx(_client(), None), {"query": "pricing"})
        assert result["ok"] is False and result["error"] == "unavailable"


@pytest.mark.unit
class TestSchemaAndRegistration:
    def test_the_description_admits_that_contents_are_not_searched(self):
        description = knowledge_tools.get_search_stored_knowledge_schema()["description"]
        lowered = description.lower()
        assert "summaries" in lowered and "descriptions" in lowered
        # The load-bearing honesty: a miss is not proof, and read_document is the way past it.
        assert "a miss does not mean" in lowered
        assert "read_document" in description

    def test_the_description_never_offers_a_handle_the_named_tool_would_refuse(self):
        schema = knowledge_tools.get_search_stored_knowledge_schema()
        blob = f"{schema['description']} {schema['parameters']}"
        # mount_file resolves only the opaque handles offered on the turn itself, so pointing a
        # stored file_id at it would spend a round on a guaranteed refusal.
        assert "mount_file" not in blob
        assert "read_document" in schema["description"]

    def test_the_description_tells_the_model_to_use_its_own_judgment(self):
        description = knowledge_tools.get_search_stored_knowledge_schema()["description"]
        assert "own judgment" in description.lower()

    def test_the_schema_requires_a_query_and_nothing_else(self):
        schema = knowledge_tools.get_search_stored_knowledge_schema()
        assert schema["name"] == "search_stored_knowledge"
        assert schema["parameters"]["required"] == ["query"]
        assert schema["parameters"]["additionalProperties"] is False

    def test_it_registers_on_both_surfaces(self):
        registry = ToolRegistry()
        knowledge_tools.register_knowledge_tools(registry)
        assert any(s["name"] == "search_stored_knowledge" for s in registry.schemas({}))
        assert any(s["name"] == "search_stored_knowledge"
                   for s in registry.schemas({}, surface="channel"))


@pytest.fixture
def real_db(tmp_path):
    """A real SQLite database — the LIKE escaping and the channel prefix are SQL behaviour, and
    a mock would only prove the strings were passed along."""
    import sqlite3

    from database import DatabaseManager

    db = DatabaseManager("test")
    db.db_path = f"{tmp_path}/knowledge.db"
    db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
    db.conn.row_factory = sqlite3.Row
    db.init_schema()
    yield db
    db.conn.close()


@pytest.mark.unit
@pytest.mark.asyncio
class TestStoreQueries:
    async def test_a_document_matches_on_its_summary_or_its_filename(self, real_db):
        real_db.save_document("C1:111.0", "budget.xlsx", "application/vnd.ms-excel",
                              summary="Wholesale pricing for the spring line.", file_id="FA")
        real_db.conn.close()

        by_summary = await real_db.search_channel_documents_async("C1", "wholesale")
        by_name = await real_db.search_channel_documents_async("C1", "budget")
        by_neither = await real_db.search_channel_documents_async("C1", "quarterly")

        assert [d["filename"] for d in by_summary] == ["budget.xlsx"]
        assert [d["filename"] for d in by_name] == ["budget.xlsx"]
        assert by_neither == []

    async def test_a_search_cannot_escape_its_channel(self, real_db):
        real_db.save_document("C1:111.0", "a.pdf", "application/pdf", summary="shared plan",
                              file_id="FA")
        real_db.save_document("C2:222.0", "b.pdf", "application/pdf", summary="shared plan",
                              file_id="FB")
        real_db.save_image_metadata("C1:111.0", "https://x.test/1.png", "screenshot",
                                    analysis="shared plan diagram")
        real_db.save_image_metadata("C2:222.0", "https://x.test/2.png", "screenshot",
                                    analysis="shared plan diagram")
        real_db.conn.close()

        docs = await real_db.search_channel_documents_async("C1", "shared plan")
        images = await real_db.search_channel_image_analyses_async("C1", "shared plan")

        assert [d["filename"] for d in docs] == ["a.pdf"]
        assert [i["url"] for i in images] == ["https://x.test/1.png"]

    async def test_a_channel_whose_id_prefixes_another_stays_separate(self, real_db):
        # "C1" must not swallow "C12": the boundary is the colon, not the bare prefix.
        real_db.save_document("C1:111.0", "one.pdf", "application/pdf", summary="report",
                              file_id="FA")
        real_db.save_document("C12:222.0", "twelve.pdf", "application/pdf", summary="report",
                              file_id="FB")
        real_db.conn.close()

        assert [d["filename"] for d in
                await real_db.search_channel_documents_async("C1", "report")] == ["one.pdf"]

    async def test_like_wildcards_in_the_query_match_themselves(self, real_db):
        real_db.save_document("C1:111.0", "hit.pdf", "application/pdf",
                              summary="error 100% of the time", file_id="FA")
        real_db.save_document("C1:222.0", "miss.pdf", "application/pdf",
                              summary="error 1004 of the time", file_id="FB")
        real_db.conn.close()

        # Unescaped, "100%" would match both rows and a lone "%" would return the whole channel.
        percent = await real_db.search_channel_documents_async("C1", "100%")
        lone_wildcard = await real_db.search_channel_documents_async("C1", "%")

        assert [d["filename"] for d in percent] == ["hit.pdf"]
        # A bare "%" is a literal percent sign, so it finds the row that HAS one — not the channel.
        assert [d["filename"] for d in lone_wildcard] == ["hit.pdf"]

    async def test_an_underscore_in_the_query_is_not_a_single_character_wildcard(self, real_db):
        real_db.save_document("C1:111.0", "exact.pdf", "application/pdf",
                              summary="see run_id in the log", file_id="FA")
        real_db.save_document("C1:222.0", "wild.pdf", "application/pdf",
                              summary="see runXid in the log", file_id="FB")
        real_db.conn.close()

        found = await real_db.search_channel_documents_async("C1", "run_id")
        assert [d["filename"] for d in found] == ["exact.pdf"]

    async def test_image_analyses_are_searched_but_prompts_are_not(self, real_db):
        real_db.save_image_metadata("C1:111.0", "https://x.test/described.png", "generated",
                                    prompt="a lighthouse at dusk", analysis="A red barn in snow.")
        real_db.conn.close()

        by_analysis = await real_db.search_channel_image_analyses_async("C1", "barn")
        by_prompt = await real_db.search_channel_image_analyses_async("C1", "lighthouse")

        assert [i["url"] for i in by_analysis] == ["https://x.test/described.png"]
        # What was ASKED for is not evidence of what the picture shows.
        assert by_prompt == []

    async def test_an_edited_images_pre_edit_description_is_still_searchable(self, real_db):
        real_db.save_image_metadata("C1:111.0", "https://x.test/edited.png", "edited",
                                    analysis="Now with a blue roof.",
                                    original_analysis="A red barn in snow.")
        real_db.conn.close()

        found = await real_db.search_channel_image_analyses_async("C1", "red barn")
        assert [i["url"] for i in found] == ["https://x.test/edited.png"]

    async def test_results_come_back_newest_first_and_bounded(self, real_db):
        for i in range(1, 6):
            real_db.save_document(f"C1:{i}00.0", f"doc{i}.pdf", "application/pdf",
                                  summary="quarterly report", file_id=f"F{i}")
        real_db.conn.execute(
            "UPDATE documents SET created_at = '2026-03-0' || id || ' 10:00:00'")
        real_db.conn.close()

        rows = await real_db.search_channel_documents_async("C1", "quarterly", limit=2)
        assert [d["filename"] for d in rows] == ["doc5.pdf", "doc4.pdf"]

    async def test_an_empty_query_or_channel_returns_nothing_rather_than_everything(self,
                                                                                   real_db):
        real_db.save_document("C1:111.0", "a.pdf", "application/pdf", summary="anything",
                              file_id="FA")
        real_db.conn.close()

        assert await real_db.search_channel_documents_async("C1", "   ") == []
        assert await real_db.search_channel_documents_async("", "anything") == []
        assert await real_db.search_channel_image_analyses_async("C1", "") == []


@pytest.mark.unit
class TestSnippets:
    def test_a_snippet_is_centred_on_the_match(self):
        text = "x" * 2000 + "MATCHME" + "y" * 2000
        snippet = knowledge_tools._snippet(text, "matchme")
        assert "MATCHME" in snippet
        assert snippet.startswith("…") and snippet.endswith("…")

    def test_a_short_text_comes_back_whole_and_unmarked(self):
        snippet = knowledge_tools._snippet("the wholesale column is blank", "wholesale")
        assert snippet == "the wholesale column is blank"

    def test_nothing_to_quote_returns_nothing_rather_than_an_empty_string(self):
        assert knowledge_tools._snippet(None, "x") is None
        assert knowledge_tools._snippet("", "x") is None
