"""F35 — the unified mountable-file catalog (images + documents behind one opaque id space)."""
import pytest

from message_processor import thread_files


class FakeDB:
    def __init__(self, images=None, docs=None, raise_images=False, raise_docs=False):
        self._images = images or []
        self._docs = docs or []
        self._raise_images = raise_images
        self._raise_docs = raise_docs

    async def find_thread_images_async(self, thread_key):
        if self._raise_images:
            raise RuntimeError("image store is down")
        return self._images

    async def get_thread_documents_async(self, thread_key):
        if self._raise_docs:
            raise RuntimeError("document store is down")
        return self._docs


def _img(row_id, url="https://files.slack.com/x/shot.png", **kw):
    base = {"id": row_id, "url": url, "image_type": "uploaded", "analysis": "a bar chart",
            "prompt": "", "created_at": "2026-07-12T10:00:00"}
    base.update(kw)
    return base


def _doc(row_id, filename="sales.csv", **kw):
    base = {"id": row_id, "filename": filename, "mime_type": "text/csv",
            "file_id": f"F{row_id}", "url_private": f"https://files.slack.com/{filename}",
            "size_bytes": 2048, "summary": "Q3 sales by region",
            "created_at": "2026-07-12T11:00:00", "metadata": None}
    base.update(kw)
    return base


@pytest.mark.unit
class TestBuildCatalog:
    async def test_unions_images_and_documents_under_one_id_space(self):
        db = FakeDB(images=[_img(1)], docs=[_doc(7)])
        entries = await thread_files.build_catalog(db, "C1:123.45")

        ids = thread_files.valid_ids(entries)
        assert "file_img_1" in ids
        assert "file_doc_7" in ids
        # The two stores must not collide: same row id, different file.
        assert thread_files.image_file_id(3) != thread_files.document_file_id(3)

    async def test_newest_first(self):
        db = FakeDB(
            images=[_img(1, created_at="2026-07-12T09:00:00")],
            docs=[_doc(2, created_at="2026-07-12T15:00:00")],
        )
        entries = await thread_files.build_catalog(db, "C1:1")
        assert entries[0]["file_id"] == "file_doc_2"

    async def test_capped(self):
        db = FakeDB(docs=[_doc(i, created_at=f"2026-07-12T{i:02d}:00:00") for i in range(1, 24)])
        entries = await thread_files.build_catalog(db, "C1:1")
        assert len(entries) == thread_files.MAX_CATALOG

    async def test_document_with_no_slack_ref_is_dropped(self):
        # Nothing to download → offering it would only produce a mount that fails.
        db = FakeDB(docs=[_doc(1, file_id=None, url_private=None)])
        entries = await thread_files.build_catalog(db, "C1:1")
        assert entries == []

    async def test_image_with_no_url_is_dropped(self):
        db = FakeDB(images=[_img(1, url=None)])
        entries = await thread_files.build_catalog(db, "C1:1")
        assert entries == []

    async def test_store_failure_degrades_to_the_other_store(self):
        # A catalog failure must cost the tool, never the turn.
        db = FakeDB(images=[_img(1)], docs=[], raise_docs=True)
        entries = await thread_files.build_catalog(db, "C1:1")
        assert [e["file_id"] for e in entries] == ["file_img_1"]

    async def test_no_db_or_thread_key_is_empty_not_an_error(self):
        assert await thread_files.build_catalog(None, "C1:1") == []
        assert await thread_files.build_catalog(FakeDB(), "") == []

    async def test_generated_documents_are_marked_generated(self):
        # A deck we published earlier is mountable again — that is what makes "revise the deck
        # you made yesterday" possible after the container is long gone.
        db = FakeDB(docs=[_doc(1, filename="deck.pptx",
                               metadata='{"source": "generated", "tool": "code_interpreter"}')])
        entries = await thread_files.build_catalog(db, "C1:1")
        assert entries[0]["origin"] == "generated"

    async def test_uploaded_is_the_default_origin(self):
        db = FakeDB(docs=[_doc(1, metadata="not json at all")])
        entries = await thread_files.build_catalog(db, "C1:1")
        assert entries[0]["origin"] == "uploaded"

    async def test_image_filename_derived_from_url(self):
        db = FakeDB(images=[_img(1, url="https://files.slack.com/T1/F2/quarterly%20chart.png")])
        entries = await thread_files.build_catalog(db, "C1:1")
        assert entries[0]["filename"] == "quarterly chart.png"


@pytest.mark.unit
class TestResolve:
    def test_only_advertised_ids_resolve(self):
        entries = [{"file_id": "file_doc_1"}]
        assert thread_files.resolve(entries, "file_doc_1") is not None
        # A syntactically valid id is not authorization.
        assert thread_files.resolve(entries, "file_doc_2") is None
        assert thread_files.resolve(entries, "file_img_1") is None
        assert thread_files.resolve(None, "file_doc_1") is None


@pytest.mark.unit
class TestCatalogLines:
    def test_lines_carry_name_type_and_description(self):
        entries = [{
            "file_id": "file_doc_1", "filename": "sales.csv", "mime_type": "text/csv",
            "size_bytes": 2048, "description": "Q3 sales by region",
        }]
        line = thread_files.catalog_lines(entries)
        assert "file_doc_1" in line
        assert "sales.csv" in line
        assert "2 KB" in line
        assert "Q3 sales by region" in line


@pytest.mark.unit
class TestDuplicateRowsCollapse:
    """One Slack file, one catalog entry.

    The same upload gets written twice in practice: the unattended catalog records it when the
    bot stays quiet, and a turn records it again if it later processes the same message.
    `save_document` is a plain INSERT, so both rows survive. Two ids for one file wastes an enum
    slot and invites the model to mount the same thing twice.
    """

    async def test_same_slack_file_appears_once(self):
        db = FakeDB(docs=[
            _doc(1, filename="sales.csv", file_id="F9", created_at="2026-07-12T10:00:00"),
            _doc(2, filename="sales.csv", file_id="F9", created_at="2026-07-12T10:05:00"),
        ])
        entries = await thread_files.build_catalog(db, "C1:1")
        assert len(entries) == 1
        # Newest row wins — it carries whatever richer metadata arrived later.
        assert entries[0]["file_id"] == "file_doc_2"

    async def test_genuinely_different_files_both_survive(self):
        db = FakeDB(docs=[_doc(1, filename="a.csv", file_id="F1"),
                          _doc(2, filename="b.csv", file_id="F2")])
        entries = await thread_files.build_catalog(db, "C1:1")
        assert len(entries) == 2
