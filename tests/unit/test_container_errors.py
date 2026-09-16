"""Code-interpreter container failure detection: dead, wedged, and neither.

A WEDGED container is the one that cost a prod build three attempts in nine seconds: its Python
kernel died of an OOM, so every later exec fails in 2-5s with a message that says nothing
("There was an issue with your request"), while `containers.retrieve()` still answers `running`.
Nothing the API tells us names the cause, so the detector narrows on shape instead — and the
tests below are the boundary it has to hold: a real 400 from a malformed request must NOT read as
a wedge, or a plain bad request would spend a sandbox replacement on itself.
"""
from unittest.mock import MagicMock

import openai

from openai_client.container_errors import (auto_container, demote_container_tools,
                                            is_container_gone, is_container_wedged)

# The message the API returns for every exec in a container whose kernel died (measured
# 2026-09-14, three probes at +0s / +60s / +180s — identical each time).
WEDGED_MESSAGE = "There was an issue with your request. Please check your inputs and try again"
WEDGED_BODY = {"type": "invalid_request_error", "code": None,
               "message": WEDGED_MESSAGE, "param": None}


def _status_error(cls, status):
    response = MagicMock()
    response.status_code = status
    response.headers = {}
    return cls(WEDGED_MESSAGE, response=response, body=WEDGED_BODY)


class TestIsContainerWedged:

    def test_the_measured_wedged_error_is_detected(self):
        exc = openai.APIError(WEDGED_MESSAGE, request=None, body=WEDGED_BODY)
        assert getattr(exc, "status_code", None) is None   # the real streaming shape
        assert is_container_wedged(exc) is True
        assert is_container_gone(exc) is False             # it is still very much alive

    def test_the_measured_non_streaming_wedge_is_a_400(self):
        """The SAME wedged container, called non-streaming, raises `BadRequestError` (measured
        2026-09-15). Excluding 400 outright — the first cut of this detector — meant the fallback
        path recovered nothing at all."""
        exc = _status_error(openai.BadRequestError, 400)
        assert exc.status_code == 400
        assert is_container_wedged(exc) is True
        assert is_container_gone(exc) is False

    def test_a_400_that_names_the_part_it_objected_to_is_a_real_bad_request(self):
        """This is what keeps the 400 widening safe: a malformed request says WHICH part was
        wrong. Replacing the sandbox over one would throw away a working container and still get
        refused, because the request would go back out just as malformed. Generic message AND a
        named param, because that is the only shape the message rule alone would let through."""
        response = MagicMock()
        response.status_code = 400
        response.headers = {}
        exc = openai.BadRequestError(
            WEDGED_MESSAGE, response=response,
            body={"type": "invalid_request_error", "code": None,
                  "message": WEDGED_MESSAGE, "param": "input[3].content[1].source"})
        assert is_container_wedged(exc) is False

    def test_a_statusless_runtime_error_carrying_the_message_is_detected(self):
        """The shape `responses.py` builds out of a `response.failed` event: no body, no status,
        the provider's message in the string."""
        exc = RuntimeError(f"Response failed: {WEDGED_MESSAGE}")
        assert is_container_wedged(exc) is True

    def test_a_dead_container_is_gone_not_wedged(self):
        """The two recoveries are different — one re-resolves, one banks and replaces — so the
        detectors must never both fire on the same error."""
        exc = Exception("Container with id 'cntr_dead' not found.")
        assert is_container_gone(exc) is True
        assert is_container_wedged(exc) is False


class TestAutoContainer:

    def test_the_auto_declaration_carries_the_configured_size(self, monkeypatch):
        """`{"type": "auto"}` alone lands the model in a 1g sandbox (probed) — the size that
        produced the wedge. The size rides on the declaration, which is why this is a factory.

        Configured to `4g`, deliberately NOT the 16g default: hardcoding the default would pass
        against a factory that ignored config entirely, which is the bug worth catching."""
        import config as config_module
        monkeypatch.setattr(config_module.config, "code_interpreter_memory_limit", "4g")

        assert auto_container() == {"type": "auto", "memory_limit": "4g"}

    def test_demoting_a_dead_container_keeps_the_size(self, monkeypatch):
        """The recovery retry must not silently drop back to 1g."""
        import config as config_module
        monkeypatch.setattr(config_module.config, "code_interpreter_memory_limit", "4g")

        tools, changed = demote_container_tools(
            [{"type": "code_interpreter", "container": "cntr_dead"}])

        assert changed is True
        assert tools[0]["container"] == {"type": "auto", "memory_limit": "4g"}
