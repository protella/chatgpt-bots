"""Test configuration for pytest."""

import pytest


def pytest_configure(config):
    """Configure pytest to run async tests."""
    config.addinivalue_line("markers", "asyncio: mark test as an asyncio coroutine")


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 