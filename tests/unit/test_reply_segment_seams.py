"""Two sentences of one reply, glued together by a hosted tool call in between.

A hosted tool (the code sandbox, web search) splits the model's answer into SEPARATE output
items: some text, the call, then more text. Neither item knows the other exists, so raw
concatenation produced this in the channel, live 2026-07-24:

    "…screenshot it at 1920×1080, same approach Claude described.Third version is built via…"

The seam is only visible at the item boundary, which is why it is inserted in the API layer —
and why the non-streaming paths, which glue items the same way, get it for free.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from openai_client.api.responses import _join_output_text, _segment_separator

pytestmark = pytest.mark.unit


def _part(text):
    return SimpleNamespace(text=text)


def _msg(*texts):
    return SimpleNamespace(type="message", content=[_part(t) for t in texts])


def _response(*items):
    return SimpleNamespace(output=list(items))


class TestSeparator:
    def test_a_finished_sentence_gets_a_paragraph_break(self):
        assert _segment_separator("same approach Claude described.") == "\n\n"

    @pytest.mark.parametrize("tail", ["done!", "really?", "and so on…", "the totals:", "one thing;"])
    def test_every_sentence_ending_counts(self, tail):
        assert _segment_separator(tail) == "\n\n"

    @pytest.mark.parametrize("tail", ['he said "no way."', "(see the chart above.)", "*ready.*",
                                      "`compute.`"])
    def test_punctuation_hiding_behind_a_closer_still_counts(self, tail):
        assert _segment_separator(tail) == "\n\n"

    def test_text_that_stopped_mid_sentence_gets_a_space(self):
        """"The total is" → sandbox → "56,088". A paragraph break here is worse than the bug."""
        assert _segment_separator("the total is") == " "

    @pytest.mark.parametrize("tail", ["already spaced ", "already broken\n", ""])
    def test_a_seam_the_model_wrote_itself_is_left_alone(self, tail):
        assert _segment_separator(tail) == ""


class TestNonStreamingJoin:
    def test_the_live_failure_case(self):
        text = _join_output_text(_response(
            _msg("Yep — I'll build a third version, same approach Claude described."),
            _msg("Third version is built via HTML/CSS and rasterized to PNG."),
        ))
        assert "described.\n\nThird version" in text
        assert "described.Third" not in text

    def test_parts_INSIDE_one_item_are_still_joined_raw(self):
        """One message split across content parts is one continuous sentence, not two."""
        assert _join_output_text(_response(_msg("North leads ", "at 65,316 units"))) == \
            "North leads at 65,316 units"

    def test_a_single_item_is_untouched(self):
        assert _join_output_text(_response(_msg("just an answer."))) == "just an answer."

    def test_items_carrying_no_text_are_skipped_without_leaving_a_seam(self):
        """Tool calls and reasoning items sit between the text items; they must not add gaps."""
        text = _join_output_text(_response(
            _msg("Computing."),
            SimpleNamespace(type="code_interpreter_call", content=None),
            SimpleNamespace(type="reasoning", content=[]),
            _msg("It's 42."),
        ))
        assert text == "Computing.\n\nIt's 42."

    def test_an_empty_response_is_empty(self):
        assert _join_output_text(_response()) == ""
        assert _join_output_text(SimpleNamespace(output=None)) == ""


def _delta(text, index=None):
    e = SimpleNamespace(type="response.output_text.delta", delta=text)
    if index is not None:
        e.output_index = index
    return e


class TestStreamingSeam:
    """The streamed path has to seam it live: the separator must reach the callback too, or
    Slack shows the glued text while the returned string is correct."""

    @pytest.fixture
    def client(self):
        with patch("openai_client.base.AsyncOpenAI"), patch("openai_client.base.aiohttp.ClientSession"):
            from openai_client import OpenAIClient
            return OpenAIClient()

    async def _run(self, client, events):
        async def _stream(*a, **k):
            for e in events:
                yield e

        chunks: list = []
        client._safe_stream_iteration = _stream
        client._safe_api_call = AsyncMock(return_value=object())

        async def _cb(chunk):
            chunks.append(chunk)

        text = await client.create_streaming_response(
            messages=[{"role": "user", "content": "x"}], stream_callback=_cb)
        return text, chunks

    @pytest.mark.asyncio
    async def test_a_new_output_item_seams_the_reply(self, client):
        text, chunks = await self._run(client, [
            SimpleNamespace(type="response.created"),
            _delta("Yep — same approach Claude ", 0),
            _delta("described.", 0),
            _delta("Third version", 2),
            _delta(" is built.", 2),
        ])

        assert text == "Yep — same approach Claude described.\n\nThird version is built."
        assert "".join(chunks) == text, "Slack must receive the separator too"

    @pytest.mark.asyncio
    async def test_one_continuous_item_is_never_seamed(self, client):
        text, _ = await self._run(client, [
            SimpleNamespace(type="response.created"),
            _delta("one ", 0), _delta("reply ", 0), _delta("only.", 0),
        ])
        assert text == "one reply only."

    @pytest.mark.asyncio
    async def test_a_mid_sentence_split_gets_a_space_not_a_break(self, client):
        text, _ = await self._run(client, [
            SimpleNamespace(type="response.created"),
            _delta("The total is", 0), _delta("56,088.", 2),
        ])
        assert text == "The total is 56,088."

    @pytest.mark.asyncio
    async def test_no_output_index_degrades_to_the_old_behaviour(self, client):
        """An API that stops sending output_index must not start seaming every delta."""
        text, _ = await self._run(client, [
            SimpleNamespace(type="response.created"), _delta("a."), _delta("b."),
        ])
        assert text == "a.b."
