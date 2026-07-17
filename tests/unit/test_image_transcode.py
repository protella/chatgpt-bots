"""F50b end-to-end: the ambient vision worker transcodes a decodable-but-unsupported image to
PNG in memory before it rides the vision call, instead of failing it as an unsupported type.

The attachment and slack-URL call sites are covered in test_image_validation.py; this pins the
third caller — the ambient worker in message_processor/ambient_memory.py — where the transcoded
bytes have to reach `analyze_images` as a real image/png data URL.
"""
import sqlite3
import tempfile
from io import BytesIO

import pytest
from PIL import Image

from database import DatabaseManager
from message_processor.ambient_memory import AmbientArtifactService, _Job

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def _bmp() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (3, 3), "red").save(buf, format="BMP")
    return buf.getvalue()


class _CapturingOpenAI:
    def __init__(self):
        self.seen_data_urls = []

    async def analyze_images(self, images, question, enhance_prompt=False, **kw):
        self.seen_data_urls.append(images[0]["image_url"])
        return "A red square."


class _Client:
    def __init__(self, data):
        self.data = data

    async def download_file(self, url, fid=None, max_bytes=None, **kw):
        return self.data


class _Pulse:
    def upsert_artifacts(self, channel_id, source_ts, notes):
        return True


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = DatabaseManager("test")
        d.db_path = f"{tmpdir}/test.db"
        d.conn = sqlite3.connect(d.db_path, check_same_thread=False, isolation_level=None)
        d.conn.row_factory = sqlite3.Row
        d.init_schema()
        yield d


async def test_ambient_worker_transcodes_bmp_before_vision(db):
    openai = _CapturingOpenAI()
    svc = AmbientArtifactService(db=db, openai_client=openai, channel_pulse=_Pulse())
    svc._client = _Client(_bmp())

    await svc._process(_Job(kind="image", channel_id="C1", source_ts="1.1", conversation_ts="1.1",
                            ref="F1", url="https://files/f1", filename="pic.bmp",
                            mimetype="image/bmp"))

    # The BMP the API rejects became a PNG the API accepts, and THAT is what the vision call saw.
    assert openai.seen_data_urls, "vision was never called — the BMP was wrongly failed"
    assert openai.seen_data_urls[0].startswith("data:image/png;base64,")

    art = (await db.get_ambient_artifacts_for_messages("C1", ["1.1"]))["1.1"][0]
    assert art["status"] == "ready" and art["summary"]
