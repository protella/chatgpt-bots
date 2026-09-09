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


class WideningDB(FakeDB):
    """A FakeDB that also answers the two CHANNEL-wide lookups.

    `find_channel_images_async` honours `within_hours` the way the real SQL does, so the
    lookback bound is exercised through the same contract production relies on rather than
    asserted against a stub that ignores it.
    """

    def __init__(self, channel_images=None, channel_docs=None, **kw):
        super().__init__(**kw)
        self._channel_images = channel_images or []
        self._channel_docs = channel_docs or []
        self.channel_calls = []

    async def find_channel_images_async(self, channel_id, within_hours=None, limit=50):
        self.channel_calls.append({"channel_id": channel_id, "within_hours": within_hours,
                                   "limit": limit})
        rows = self._channel_images
        if within_hours is not None:
            cutoff = _ago(hours=within_hours)
            rows = [r for r in rows if str(r.get("created_at") or "") >= cutoff]
        # The real lookup is NEWEST first.
        return sorted(rows, key=lambda r: str(r.get("created_at") or ""), reverse=True)[:limit]

    async def get_channel_documents_async(self, channel_id):
        # The real lookup is oldest-first and carries no time bound.
        return sorted(self._channel_docs, key=lambda r: str(r.get("created_at") or ""))


def _ago(*, hours=0, minutes=0):
    """A timestamp shaped like SQLite's CURRENT_TIMESTAMP, that far in the past."""
    from datetime import datetime, timedelta, timezone
    stamp = datetime.now(timezone.utc) - timedelta(hours=hours, minutes=minutes)
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


@pytest.mark.unit
class TestDMWidening:
    """A DM is one conversation; Slack just splits it into roots.

    Live 2026-09-09: four images generated minutes earlier in the same DM sat under a different
    root than "now build me a deck from those", so the catalog was empty, `mount_file` had
    nothing to offer, and the build job saw an empty /mnt/data. Channels stay strict — there a
    thread is a real conversation boundary, not an accident of the surface.
    """

    async def test_an_image_from_another_root_of_this_dm_is_mountable(self):
        db = WideningDB(channel_images=[_img(7, url="https://files.slack.com/x/deck1.png",
                                             created_at=_ago(minutes=5))])
        entries = await thread_files.build_catalog(db, "D0BKX77NU66:1784925818.611379")

        assert thread_files.valid_ids(entries) == ["file_img_7"]
        assert entries[0]["scope"] == "earlier in this DM"
        assert "[earlier in this DM]" in thread_files.catalog_lines(entries)
        # The whole point: mount_file's executor resolves the id against this snapshot.
        assert thread_files.resolve(entries, "file_img_7") is not None
        # One DM, and only this one.
        assert db.channel_calls == [{"channel_id": "D0BKX77NU66",
                                     "within_hours": thread_files.DM_LOOKBACK_HOURS,
                                     "limit": thread_files.MAX_CATALOG * 2}]

    async def test_a_document_from_another_root_of_this_dm_is_mountable(self):
        db = WideningDB(channel_docs=[_doc(4, filename="q3.csv", file_id="F4",
                                           created_at=_ago(hours=2))])
        entries = await thread_files.build_catalog(db, "D0BKX77NU66:1784925818.611379")

        assert thread_files.valid_ids(entries) == ["file_doc_4"]
        assert entries[0]["scope"] == "earlier in this DM"

    async def test_a_channel_gets_nothing_extra(self):
        db = WideningDB(channel_images=[_img(7, url="https://files.slack.com/x/deck1.png",
                                             created_at=_ago(minutes=5))],
                        channel_docs=[_doc(4, file_id="F4", created_at=_ago(hours=2))])
        entries = await thread_files.build_catalog(db, "C0BKX77NU66:1784925818.611379")

        assert entries == []
        assert db.channel_calls == []

    async def test_anything_older_than_the_lookback_is_left_alone(self):
        db = WideningDB(
            channel_images=[_img(7, url="https://files.slack.com/x/old.png",
                                 created_at=_ago(hours=thread_files.DM_LOOKBACK_HOURS + 6))],
            channel_docs=[_doc(4, file_id="F4", created_at=_ago(hours=thread_files.DM_LOOKBACK_HOURS + 6))],
        )
        entries = await thread_files.build_catalog(db, "D0BKX77NU66:1")
        assert entries == []

    async def test_a_file_already_in_the_strict_catalog_is_not_offered_twice(self):
        url = "https://files.slack.com/x/deck1.png"
        db = WideningDB(images=[_img(7, url=url)],
                        channel_images=[_img(7, url=url, created_at=_ago(minutes=5))])
        entries = await thread_files.build_catalog(db, "D0BKX77NU66:1")

        assert thread_files.valid_ids(entries) == ["file_img_7"]
        # The STRICT entry survives, unmarked: it is this thread's own file.
        assert "scope" not in entries[0]

    async def test_strict_entries_win_the_cap_and_are_unchanged(self):
        strict = [_doc(i, filename=f"s{i}.csv", file_id=f"F{i}",
                       created_at=f"2026-09-09T{i:02d}:00:00")
                  for i in range(1, thread_files.MAX_CATALOG + 1)]
        db = WideningDB(docs=strict,
                        channel_images=[_img(99, url="https://files.slack.com/x/late.png",
                                             created_at=_ago(minutes=1))])
        entries = await thread_files.build_catalog(db, "D0BKX77NU66:1")

        assert len(entries) == thread_files.MAX_CATALOG
        assert "file_img_99" not in thread_files.valid_ids(entries)
        assert all("scope" not in e for e in entries)

    async def test_a_broken_widening_still_yields_the_strict_catalog(self):
        db = WideningDB(docs=[_doc(1)])

        async def boom(*a, **kw):
            raise RuntimeError("channel store is down")

        db.find_channel_images_async = boom
        db.get_channel_documents_async = boom
        entries = await thread_files.build_catalog(db, "D0BKX77NU66:1")
        assert thread_files.valid_ids(entries) == ["file_doc_1"]

    async def test_a_recent_document_outranks_a_wall_of_images(self):
        """The room is shared, and recency decides who gets it.

        Filling from images first and asking documents for the leftovers meant a DM holding a
        screenshot-per-message could never offer the CSV uploaded a minute ago: the cap was
        spent before the document query ran.
        """
        db = WideningDB(
            channel_images=[_img(i, url=f"https://files.slack.com/x/shot{i}.png",
                                 created_at=_ago(hours=2))
                            for i in range(1, 21)],
            channel_docs=[_doc(4, filename="q3.csv", file_id="F4",
                               created_at=_ago(hours=1))],
        )
        entries = await thread_files.build_catalog(db, "D0BKX77NU66:1")

        assert len(entries) == thread_files.MAX_CATALOG
        ids = thread_files.valid_ids(entries)
        assert "file_doc_4" in ids, "the newest file must not be crowded out by older images"
        # Newest first, and the document IS the newest.
        assert ids[0] == "file_doc_4"
