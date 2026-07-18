"""F39 — logger degrades gracefully when concurrent_log_handler is missing.

When concurrent_log_handler is not installed, USE_CONCURRENT_HANDLER is False at import time, so the
RotatingFileHandler fallback fires during the module-level `main_logger = setup_logger("slack_bot")`
— i.e. while `main_logger` is still unbound. The fallback branch used to reference `main_logger`
there, raising NameError and taking down the whole import. It must warn (via `warnings`) and carry
on instead.
"""
import builtins
import importlib
import sys
import warnings

import pytest


@pytest.mark.unit
def test_import_survives_missing_concurrent_log_handler():
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "concurrent_log_handler":
            raise ImportError("simulated: concurrent_log_handler not installed")
        return real_import(name, *args, **kwargs)

    try:
        builtins.__import__ = blocking_import
        sys.modules.pop("concurrent_log_handler", None)
        # reload() keeps the module's existing globals, so drop the already-bound main_logger to
        # reproduce the real from-scratch state where the fallback fires before it is assigned.
        if "logger" in sys.modules:
            sys.modules["logger"].__dict__.pop("main_logger", None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Reload runs the module body, including `main_logger = setup_logger("slack_bot")`,
            # which hits the fallback branch — the exact F39 NameError window.
            reloaded = importlib.reload(importlib.import_module("logger"))

        assert reloaded.USE_CONCURRENT_HANDLER is False
        # Import completed: main_logger got bound instead of raising NameError.
        assert reloaded.main_logger is not None
        assert any("ConcurrentRotatingFileHandler" in str(w.message) for w in caught), \
            "the fallback should warn about the degraded handler"
    finally:
        # Restore the real module state so the rest of the suite sees a normal logger.
        builtins.__import__ = real_import
        importlib.reload(importlib.import_module("logger"))
