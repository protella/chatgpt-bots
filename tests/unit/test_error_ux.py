"""Error-UX fixes — one formatter, no raw-exception leaks, loud failure for
silently-accepted work (attachments, queued messages, button clicks).

Principles under test: "Two classes, two voices", "The indicator always
resolves", "Anything silently accepted must be loudly failed", "One formatter".
"""
from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


from message_processor.client_contract import Message
from main import ChatBotV2
from message_processor.base import MessageProcessor
from message_processor.utilities import MessageUtilitiesMixin
from slack_client.event_handlers.settings import SlackSettingsHandlersMixin
from slack_client.formatting.text import SlackFormattingMixin


# --------------------------------------------------------------- one formatter

class _Formatter(SlackFormattingMixin):
    pass


class TestFormatErrorMessage:
    fmt = _Formatter()

    def test_authored_copy_passes_through(self):
        msg = "⏱️ **Taking Too Long**\n\nOpenAI is being slow right now.\n\nPlease try again in a moment."
        assert self.fmt.format_error_message(msg) == msg

    def test_authored_colon_emoji_passes_through(self):
        msg = ":warning: **Couldn't Load Conversation History**\n\nYour message wasn't processed — please try again."
        assert self.fmt.format_error_message(msg) == msg

    def test_authored_single_asterisk_passes_through(self):
        msg = "⚠️ *Unsupported File Type*\n\nI noticed you uploaded: *x.bin*"
        assert self.fmt.format_error_message(msg) == msg

    def test_generic_message_is_idempotent(self):
        g = self.fmt.GENERIC_ERROR_MESSAGE
        assert self.fmt.format_error_message(g) == g

    def test_raw_exception_never_leaks(self):
        raw = "KeyError: 'C123:1719000000.123456'"
        out = self.fmt.format_error_message(raw)
        assert "C123" not in out
        assert out == self.fmt.GENERIC_ERROR_MESSAGE

    def test_openai_error_blob_never_leaks(self):
        raw = ("Error code: 429 - {'error': {'message': 'Rate limit reached for "
               "gpt-5.5 on tokens per min', 'type': 'tokens'}}")
        out = self.fmt.format_error_message(raw)
        assert "Rate limit reached" not in out
        assert "Error code" not in out
        assert out == self.fmt.GENERIC_ERROR_MESSAGE

    def test_no_scaffold_ever(self):
        # The old *Error Code:* / *Type:* / ```Details``` template is retired.
        for raw in ("timeout while reading", "⏱️ **Taking Too Long**\n\nRetry."):
            out = self.fmt.format_error_message(raw)
            assert "*Error Code:*" not in out
            assert "```" not in out

    def test_empty_input_gets_generic(self):
        assert self.fmt.format_error_message("") == self.fmt.GENERIC_ERROR_MESSAGE


# ------------------------------------------------- outer catch-all (findings 1+7)

def _bot_and_client(process_error, delete_error=None):
    from message_processor.stale_send_guard import ConversationWatermarks
    # A real watermark owner: handle_message opens a lease on its first line, and a stand-in
    # host without one would be testing a turn that skipped the guard entirely.
    bot = SimpleNamespace(processor=MagicMock(), watermarks=ConversationWatermarks())
    bot.processor.process_message = AsyncMock(side_effect=process_error)
    bot.processor.thread_manager = None  # already_processing peek short-circuits
    client = MagicMock()
    client.send_thinking_indicator = AsyncMock(return_value="T1")
    client.update_message = AsyncMock()
    client.delete_message = AsyncMock(side_effect=delete_error)
    client.handle_error = AsyncMock()
    return bot, client


def _msg(**meta):
    return Message(text="hi", user_id="U1", channel_id="C1", thread_id="1.0",
                   metadata={"ts": "1.0", **meta})


class TestOuterCatchAll:
    async def test_raw_exception_not_sent_to_user(self):
        bot, client = _bot_and_client(RuntimeError("secret-internal-detail /tmp/x"))
        await ChatBotV2.handle_message(bot, _msg(), client)
        assert client.handle_error.await_count == 1
        sent = client.handle_error.await_args.args[2]
        assert "secret-internal-detail" not in sent
        assert "Something Went Wrong" in sent

    async def test_failed_indicator_delete_does_not_skip_error_notice(self):
        bot, client = _bot_and_client(RuntimeError("boom"), delete_error=Exception("delete died"))
        await ChatBotV2.handle_message(bot, _msg(), client)
        assert client.handle_error.await_count == 1


# ----------------------------------------------- image-gen delete awaited (finding 3)

def test_image_gen_delete_message_calls_are_awaited():
    """AST guard: every client.delete_message(...) in image_gen.py sits under await."""
    import message_processor.handlers.image_gen as mod
    with open(mod.__file__) as f:
        tree = ast.parse(f.read())
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "delete_message"]
    assert calls, "expected at least one delete_message call site"
    for call in calls:
        assert isinstance(parents[call], ast.Await), \
            f"un-awaited delete_message at line {call.lineno}"


# ------------------------------------------- attachment download failures (finding 4)

def _utils_self():
    fake = SimpleNamespace()
    fake.log_debug = MagicMock()
    fake.log_info = MagicMock()
    fake.log_warning = MagicMock()
    fake.log_error = MagicMock()
    fake.db = None
    fake.document_handler = None
    fake.image_url_handler = MagicMock()
    fake.image_url_handler.max_image_size = 20 * 1024 * 1024
    fake.image_url_handler.process_urls_from_text = AsyncMock(return_value=([], []))
    fake._extract_slack_file_urls = MagicMock(return_value=[])
    return fake


class TestDownloadFailuresSurface:
    async def test_failed_image_attachment_download_reported(self):
        fake = _utils_self()
        client = MagicMock()
        client.download_file = AsyncMock(return_value=None)
        msg = Message(text="", user_id="U1", channel_id="C1", thread_id="1.0",
                      attachments=[{"type": "image", "name": "photo.png",
                                    "mimetype": "image/png", "id": "F1", "url": "u"}])
        _, _, unsupported = await MessageUtilitiesMixin._process_attachments(fake, msg, client)
        assert [f for f in unsupported
                if f["name"] == "photo.png" and f.get("error") == "download_failed"]

    async def test_failed_document_attachment_download_reported(self):
        fake = _utils_self()
        fake.document_handler = MagicMock()
        fake.document_handler.is_document_file = MagicMock(return_value=True)
        client = MagicMock()
        client.download_file = AsyncMock(return_value=None)
        msg = Message(text="", user_id="U1", channel_id="C1", thread_id="1.0",
                      attachments=[{"type": "file", "name": "report.pdf",
                                    "mimetype": "application/pdf", "id": "F2", "url": "u"}])
        _, _, unsupported = await MessageUtilitiesMixin._process_attachments(fake, msg, client)
        assert [f for f in unsupported
                if f["name"] == "report.pdf" and f.get("error") == "download_failed"]

    async def test_failed_external_url_download_reported(self):
        fake = _utils_self()
        fake.image_url_handler.process_urls_from_text = AsyncMock(
            return_value=([], ["http://example.com/img.png"]))
        client = MagicMock()
        msg = Message(text="look at http://example.com/img.png", user_id="U1",
                      channel_id="C1", thread_id="1.0")
        _, _, unsupported = await MessageUtilitiesMixin._process_attachments(fake, msg, client)
        assert [f for f in unsupported
                if f["name"] == "http://example.com/img.png"
                and f.get("error") == "download_failed"]

    async def test_failed_slack_url_download_reported(self):
        fake = _utils_self()
        fake._extract_slack_file_urls = MagicMock(
            return_value=["https://files.slack.com/files-pri/T1-F1/report.pdf"])

        class SlackBot:  # name checked via client.__class__.__name__
            pass
        client = SlackBot()
        client.download_file = AsyncMock(return_value=None)
        client.extract_file_id_from_url = MagicMock(return_value=None)
        msg = Message(text="see https://files.slack.com/files-pri/T1-F1/report.pdf",
                      user_id="U1", channel_id="C1", thread_id="1.0")
        _, _, unsupported = await MessageUtilitiesMixin._process_attachments(fake, msg, client)
        assert [f for f in unsupported
                if f["name"] == "report.pdf" and f.get("error") == "download_failed"]


class TestFailedFilesNotice:
    build = staticmethod(MessageProcessor._build_failed_files_notice)

    def test_download_failures_get_reupload_copy(self):
        out = self.build([{"name": "report.pdf", "type": "file",
                           "mimetype": "unknown", "error": "download_failed"}])
        assert "Couldn't Download File" in out
        assert "report.pdf" in out
        assert "Unsupported File Type" not in out

    def test_unsupported_keeps_format_explainer(self):
        out = self.build([{"name": "x.bin", "type": "file", "mimetype": "application/octet-stream"}])
        assert "Unsupported File Type" in out
        assert "Currently supported" in out
        assert "Couldn't Download File" not in out

    def test_mixed_shows_both_sections(self):
        out = self.build([
            {"name": "a.pdf", "type": "file", "mimetype": "unknown", "error": "download_failed"},
            {"name": "b.bin", "type": "file", "mimetype": "application/octet-stream"},
        ])
        assert "Couldn't Download File" in out
        assert "Unsupported File Type" in out


# --------------------------------------------- queued-batch drain failure (finding 5)

class TestDrainFailureNotice:
    def _fake(self):
        fake = SimpleNamespace()
        fake.thread_manager = MagicMock()
        fake.log_error = MagicMock()
        return fake

    async def test_notice_posted_and_refresh_flagged(self):
        fake = self._fake()
        client = MagicMock()
        client.send_message_async = AsyncMock()
        msg = _msg()
        await MessageProcessor._notify_drain_failure(fake, msg, client, "C1:1.0")
        fake.thread_manager.mark_needs_refresh.assert_called_once_with("C1:1.0")
        assert client.send_message_async.await_count == 1
        text = client.send_message_async.await_args.args[2]
        assert "catching up" in text and "re-send" in text

    async def test_notice_failure_is_swallowed(self):
        fake = self._fake()
        client = MagicMock()
        client.send_message_async = AsyncMock(side_effect=Exception("slack down"))
        await MessageProcessor._notify_drain_failure(fake, _msg(), client, "C1:1.0")
        fake.log_error.assert_called()  # logged, never raised


# ------------------------------------------------- footer modal ephemeral (finding 6)

class _FakeApp:
    def __init__(self):
        self.handlers = {}

    def _reg(self, kind, key):
        def deco(fn):
            self.handlers[(kind, str(key))] = fn
            return fn
        return deco

    def action(self, key): return self._reg("action", key)
    def command(self, key): return self._reg("command", key)
    def view(self, key): return self._reg("view", key)
    def event(self, key): return self._reg("event", key)
    def options(self, key): return self._reg("options", key)
    def shortcut(self, key): return self._reg("shortcut", key)


class _SettingsBot(SlackSettingsHandlersMixin):
    def __init__(self):
        self.app = _FakeApp()
        self.db = MagicMock()
        self.settings_modal = MagicMock()

    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


class TestFooterModalFailureEphemeral:
    async def test_modal_failure_sends_ephemeral(self):
        bot = _SettingsBot()
        bot.db.get_channel_settings_async = AsyncMock(side_effect=Exception("db down"))
        bot._register_settings_handlers()
        handler = bot.app.handlers[("action", "open_channel_settings")]
        client = MagicMock()
        client.chat_postEphemeral = AsyncMock()
        body = {"trigger_id": "t1", "container": {"channel_id": "C1"},
                "user": {"id": "U1"}}
        await handler(ack=AsyncMock(), body=body, client=client)
        assert client.chat_postEphemeral.await_count == 1
        kwargs = client.chat_postEphemeral.await_args.kwargs
        assert kwargs["channel"] == "C1" and kwargs["user"] == "U1"

    async def test_ephemeral_failure_is_swallowed(self):
        bot = _SettingsBot()
        bot.db.get_channel_settings_async = AsyncMock(side_effect=Exception("db down"))
        bot._register_settings_handlers()
        handler = bot.app.handlers[("action", "open_channel_settings")]
        client = MagicMock()
        client.chat_postEphemeral = AsyncMock(side_effect=Exception("also down"))
        body = {"trigger_id": "t1", "container": {"channel_id": "C1"},
                "user": {"id": "U1"}}
        await handler(ack=AsyncMock(), body=body, client=client)  # must not raise


# ------------------------- a malformed record is not a Slack outage (FX1 F7 / codex P2 review)

class TestStreamTimestampNotice:
    """`StreamTimestampError` is a sibling of HistoryFetchError, not a flavour of it.

    Both fail a channel turn closed, but they are opposite failures: Slack being busy is
    transient and worth retrying, while a timestamp that does not parse is a malformed record and
    will not parse on the next attempt either. Wearing the Slack notice would send the user back
    for a second helping of the same failure ("please try again in a moment"), which is the exact
    dishonesty the four-code split exists to avoid.
    """

    def test_it_gets_its_own_code_and_says_retrying_will_not_help(self):
        from message_processor import channel_stream

        code, notice = MessageProcessor._channel_stream_failure(
            channel_stream.StreamTimestampError("unparseable timestamp: 'x'"))
        assert code == "stream_data_invalid"
        assert "won't clear this one" in notice["message"]
        assert notice["status"] and not notice["status"].startswith("Something")
        # No raw exception text, and no invented cause.
        assert "unparseable timestamp: 'x'" not in notice["message"]

    def test_it_does_not_borrow_the_slack_history_notice(self):
        from message_processor import channel_stream

        _, invalid = MessageProcessor._channel_stream_failure(
            channel_stream.StreamTimestampError("boom"))
        _, fetch_failed = MessageProcessor._channel_stream_failure(
            channel_stream.ChannelStreamError("boom"))
        assert invalid["message"] != fetch_failed["message"]
        assert "try again in a moment" in fetch_failed["message"]
        assert "try again in a moment" not in invalid["message"]

    def test_the_history_fetch_default_still_owns_every_other_condition(self):
        from message_processor import channel_stream

        code, _ = MessageProcessor._channel_stream_failure(
            channel_stream.ChannelStreamError("boom"))
        assert code == "history_fetch_failed"


# --------------------- an over-size refusal from the API is an over-budget turn (codex r2 item 4)

class TestChannelOversizeBackstop:
    """The admission estimate is supposed to catch this at the door. When it does not, the API's
    refusal must arrive as the same honest "too much for one request" notice — not as "something
    went wrong", and never as a compaction retry: a channel request IS the pinned window, so
    trimming it would answer a different question than the one that was admitted.
    """

    @pytest.mark.parametrize("error", [
        Exception("Request too large for gpt-5.6-sol"),
        Exception("400 - Invalid 'input': string too long"),
        Exception("This model's maximum context length is 1050000 tokens"),
        SimpleNamespace(code="string_above_max_length"),
        SimpleNamespace(status_code=413),
    ])
    def test_every_wording_of_a_size_refusal_is_recognized(self, error):
        assert MessageProcessor._channel_request_too_large(error) is True

    @pytest.mark.parametrize("error", [
        Exception("mcp server 424 failed"),
        Exception("rate limit exceeded"),
        Exception("context7 server is down"),
        SimpleNamespace(status_code=500),
    ])
    def test_unrelated_failures_are_not_mistaken_for_size(self, error):
        assert MessageProcessor._channel_request_too_large(error) is False

    def test_the_notice_it_maps_to_is_the_over_budget_one(self):
        from message_processor import channel_stream

        code, notice = MessageProcessor._channel_stream_failure(
            channel_stream.StreamOverBudgetError("C1: too large"))
        assert code == "stream_over_budget"
        assert "larger than I can send in one go" in notice["message"]


class TestTurnEffectsUnsettledStreamingBoundary:
    """[codex P3 r5] The outer streaming boundary must pass TurnEffectsUnsettled OUT, exactly as
    it does StaleSendSuppressed — the generic handler below it retries non-streaming, which would
    run a second attempt behind unknown, unrevoked effect state."""

    def test_the_boundary_reraises_before_the_generic_retry(self):
        import inspect

        from message_processor.handlers import text as text_mod

        src = inspect.getsource(text_mod)
        anchor = src.index("THE outer streaming boundary")
        window = src[anchor:anchor + 2000]
        stale = window.index("raise")
        unsettled = window.index("except TurnEffectsUnsettled:")
        generic = window.index("except Exception as e:")
        assert stale < unsettled < generic, (
            "TurnEffectsUnsettled must be re-raised at the outer streaming boundary, "
            "before the generic except that launches the non-streaming fallback")
        assert "raise" in window[unsettled:generic]
