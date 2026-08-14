"""The fetch primitives the shallow window is built on (SHALLOW_STREAM_RESPEC §4.3).

Two things are being pinned here, and both exist because the shallow window needs a fetch shape
the landed one could not express:

  * **THE PAGE-CEILING SENTINEL.** `None` already meant "read config", so the reply fan-out and
    the origin fetch — which must have NO page bound at all — had no way to say so. Three states
    need three representations, and a sentinel is the only way to get a third out of a parameter
    whose absent value is already spoken for.
  * **`iter_pages`.** The landed `page_messages` BUFFERS the whole window before returning, so a
    walk that must stop as soon as it has proved enough roots cannot use it: by the time it
    returns, the pages it should have stopped before are already fetched and paid for. The loop
    is EXPOSED rather than rewritten, and `page_messages` becomes its drain — which is exactly
    what makes the parity test below meaningful.
"""
from __future__ import annotations

import pytest

from message_processor.client_contract import HistoryFetchError
from config import config
from slack_client.history_fetch import (HistoryPageInvalid, FetchBudget, _USE_CONFIG_CEILING,
                                        iter_pages, page_messages)


def _page(messages, *, cursor="", has_more=False):
    page = {"ok": True, "messages": list(messages)}
    if has_more:
        page["has_more"] = True
    if cursor:
        page["response_metadata"] = {"next_cursor": cursor}
    return page


def _method(pages):
    """A conversations.history stand-in that walks `pages` by cursor, recording its calls."""
    calls = []

    async def method(**kwargs):
        calls.append(kwargs)
        index = 0
        if kwargs.get("cursor"):
            index = int(kwargs["cursor"])
        return pages[index]

    method.calls = calls
    return method


# ------------------------------------------------------------------ T67: the three states

def test_the_page_ceiling_sentinel_has_three_states():
    """T67. Omitted / None / int, and the sentinel is what makes them distinguishable."""
    # OMITTED -> config. Byte-identical to every landed caller, which passes nothing.
    assert FetchBudget()._ceiling == config.history_page_ceiling
    # The sentinel is a module-level object, so a caller can name it and this can assert
    # identity — a typed default would be a value somebody could pass by accident.
    assert FetchBudget(page_ceiling=_USE_CONFIG_CEILING)._ceiling == config.history_page_ceiling
    # EXPLICIT None -> unbounded.
    assert FetchBudget(page_ceiling=None)._ceiling is None
    # An int -> exactly that.
    assert FetchBudget(page_ceiling=3)._ceiling == 3


async def test_an_unbounded_budget_still_counts_pages_and_still_checks_the_clock():
    """`None` removes the CEILING, not the accounting: `pages_used` is reported per component,
    and the deadline is the only thing still bounding it."""
    budget = FetchBudget(page_ceiling=None, total_seconds=60)
    for _ in range(200):
        await budget.charge_page()
    assert budget.pages_used == 200          # counted…
    budget.check_deadline()                  # …and refused nothing

    spent = FetchBudget(page_ceiling=None, total_seconds=0)
    with pytest.raises(HistoryFetchError, match="budget"):
        spent.check_deadline()


async def test_an_int_ceiling_refuses_at_exactly_that_page():
    budget = FetchBudget(page_ceiling=2)
    await budget.charge_page()
    await budget.charge_page()
    with pytest.raises(HistoryFetchError, match="2-page ceiling"):
        await budget.charge_page()


# ------------------------------------------------------------------ the shared deadline

def test_total_seconds_gives_each_budget_its_own_window():
    """THE DEFECT `deadline_at` EXISTS TO FIX. Three budgets built with the same `total_seconds`
    record three different start instants, so a turn could spend three times the budget."""
    now = [100.0]
    clock = lambda: now[0]                                          # noqa: E731

    first = FetchBudget(total_seconds=60, clock=clock)
    now[0] = 130.0                                                  # 30s of history walking
    second = FetchBudget(total_seconds=60, clock=clock)

    # Same wall-clock instant, two different answers: the first budget has spent 30 of its 60,
    # the second is still whole. A turn built this way can spend 60 + 60 + 60.
    assert first.remaining_seconds() == 30.0
    assert second.remaining_seconds() == 60.0
    assert first.deadline_at is None and second.deadline_at is None


def test_deadline_at_is_absolute_shared_and_readable():
    """When it is set `total_seconds` is IGNORED, and the value is PUBLIC — the one-deadline rule
    is only enforceable at the boundary that can read it back off a component's result."""
    now = 1000.0
    shared = now + 60
    budgets = [FetchBudget(deadline_at=shared, total_seconds=5, clock=lambda: now)
               for _ in range(3)]
    assert {b.deadline_at for b in budgets} == {shared}
    # total_seconds=5 is ignored; all three see the same 60s remaining.
    assert {b.remaining_seconds() for b in budgets} == {60.0}


# ------------------------------------------------------------------ T77: iter_pages parity

async def test_page_messages_drains_iter_pages_with_parity():
    """T77. `page_messages` returns byte-identical results to draining `iter_pages` over the same
    fixture — because it IS that drain. A reimplementation that drifted would show up here."""
    pages = [_page([{"ts": "1.0"}, {"ts": "2.0"}], cursor="1", has_more=True),
             _page([{"ts": "3.0"}], cursor="2", has_more=True),
             _page([{"ts": "4.0"}])]

    drained = await page_messages(_method(pages), channel_id="C1")
    collected = []
    async for batch in iter_pages(_method(pages), channel_id="C1"):
        collected.extend(batch)
    assert drained == collected
    assert [m["ts"] for m in drained] == ["1.0", "2.0", "3.0", "4.0"]


@pytest.mark.parametrize("pages,expected,match", [
    # has_more with no cursor to follow
    ([_page([{"ts": "1.0"}], has_more=True)], HistoryFetchError, "no cursor"),
    # an empty page that still claims more
    ([_page([], cursor="1", has_more=True)], HistoryFetchError, "empty page"),
    # a shape that is not a page at all
    ([{"ok": True}], HistoryPageInvalid, "no messages field"),
])
async def test_the_three_anomalies_fail_identically_through_both(pages, expected, match):
    """Every anomaly RAISES rather than being yielded, and both entry points raise the SAME type
    with the SAME message. A consumer that breaks out early can never suppress one, because the
    raise happens while PRODUCING the page it is about to receive."""
    with pytest.raises(expected) as drained_error:
        await page_messages(_method(pages), channel_id="C1")

    with pytest.raises(expected) as iterated_error:
        async for _ in iter_pages(_method(pages), channel_id="C1"):
            pass

    assert match in str(drained_error.value)
    assert str(drained_error.value) == str(iterated_error.value)


async def test_a_consumer_that_stops_early_stops_asking():
    """The whole point of exposing the loop: a caller that has seen enough simply stops, and the
    pages it never asked for are never fetched or paid for."""
    pages = [_page([{"ts": "1.0"}], cursor="1", has_more=True),
             _page([{"ts": "2.0"}], cursor="2", has_more=True),
             _page([{"ts": "3.0"}])]

    method = _method(pages)
    seen = []
    async for batch in iter_pages(method, channel_id="C1"):
        seen.extend(batch)
        if seen:
            break
    assert len(method.calls) == 1, "an early break must not fetch the pages it skipped"

    full = _method(pages)
    await page_messages(full, channel_id="C1")
    assert len(full.calls) == 3


async def test_iter_pages_follows_the_cursor_identically(monkeypatch):
    """The cursor is threaded exactly as the landed loop threaded it, and the window parameters
    ride every page."""
    pages = [_page([{"ts": "1.0"}], cursor="1", has_more=True), _page([{"ts": "2.0"}])]
    method = _method(pages)
    await page_messages(method, channel_id="C1", oldest="0.5", latest="9.0", inclusive=True)

    assert method.calls[0]["channel"] == "C1"
    assert method.calls[0]["oldest"] == "0.5" and method.calls[0]["latest"] == "9.0"
    assert method.calls[0]["inclusive"] is True
    assert "cursor" not in method.calls[0]
    assert method.calls[1]["cursor"] == "1"
    # The window parameters ride the SECOND page too — a cursor page that dropped them would
    # silently widen the window.
    assert method.calls[1]["oldest"] == "0.5" and method.calls[1]["latest"] == "9.0"


async def test_both_entry_points_charge_the_same_budget():
    """The ceiling bounds the COMPONENT, whichever entry point spends it."""
    pages = [_page([{"ts": "1.0"}], cursor="1", has_more=True),
             _page([{"ts": "2.0"}], cursor="2", has_more=True),
             _page([{"ts": "3.0"}])]

    budget = FetchBudget(page_ceiling=2)
    with pytest.raises(HistoryFetchError, match="2-page ceiling"):
        await page_messages(_method(pages), channel_id="C1", budget=budget)

    unbounded = FetchBudget(page_ceiling=None)
    collected = []
    async for batch in iter_pages(_method(pages), channel_id="C1", budget=unbounded):
        collected.extend(batch)
    assert len(collected) == 3
    assert unbounded.pages_used == 3


async def test_an_anomaly_raises_even_when_the_consumer_breaks_immediately():
    """T77's early-break case. THE ORDERING IS THE GUARANTEE.

    A consumer that breaks after the first page never resumes the generator, so a check placed
    after the `yield` is skipped by generator cleanup and the promised exception never runs. The
    shallow history walk breaks exactly like this the moment it has proved enough roots, so this
    is the real consumer shape — not a contrived one.

    Mutation-check: moving the continuation validation back after the `yield` makes every case
    below pass silently, which is the defect this test exists to catch.
    """
    for pages, expected, match in (
            ([_page([{"ts": "1.0"}], has_more=True)], HistoryFetchError, "no cursor"),
            ([_page([], cursor="1", has_more=True)], HistoryFetchError, "empty page")):
        with pytest.raises(expected) as raised:
            async for _batch in iter_pages(_method(pages), channel_id="C1"):
                break          # the early stop, before the generator could validate afterwards
        assert match in str(raised.value)

    # And a SOUND page still yields to a consumer that stops after it — the validation must not
    # have turned every early break into a failure.
    sound = [_page([{"ts": "1.0"}], cursor="1", has_more=True), _page([{"ts": "2.0"}])]
    method = _method(sound)
    seen = []
    async for batch in iter_pages(method, channel_id="C1"):
        seen.extend(batch)
        break
    assert [m["ts"] for m in seen] == ["1.0"]
    assert len(method.calls) == 1
