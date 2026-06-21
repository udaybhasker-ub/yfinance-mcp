"""Tests for _get_ticker single-flight coalescing behaviour.

These tests verify that when multiple coroutines call ``_get_ticker`` for the
same symbol concurrently:

* Only ONE ``yf.Ticker`` construction is sent to Yahoo Finance.
* All callers receive the same Ticker object.
* A subsequent call (after the in-flight completes) is served from the TTL
  cache — still only one Yahoo call total.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from yfmcp.yf_runner import _get_ticker
from yfmcp.yf_runner import _ticker_cache
from yfmcp.yf_runner import _ticker_inflight


@pytest.fixture(autouse=True)
def clear_state() -> None:
    """Reset cache and inflight registry before every test."""
    _ticker_cache.clear_sync()
    _ticker_inflight.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ticker_mock(name: str = "FAKE") -> MagicMock:
    t = MagicMock()
    t.ticker = name
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("yfmcp.yf_runner.asyncio.to_thread")
async def test_concurrent_calls_fire_yahoo_once(mock_to_thread: AsyncMock) -> None:
    """Four simultaneous _get_ticker('MU') calls → exactly one Yahoo round-trip."""
    ticker_obj = _make_ticker_mock("MU")

    # Use an Event to hold the fetch mid-flight so all four callers pile up
    # before the result resolves.
    release = asyncio.Event()
    call_count = 0

    async def slow_to_thread(func, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        await release.wait()   # block until we release
        return func(*args, **kwargs)

    mock_to_thread.side_effect = slow_to_thread

    with patch("yfmcp.yf_runner.yf.Ticker", return_value=ticker_obj):
        # Launch four concurrent callers.
        tasks = [asyncio.create_task(_get_ticker("MU")) for _ in range(4)]

        # Give the event loop a tick so all four tasks start and the first one
        # registers its Future before the others check _ticker_inflight.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Now release the single in-flight fetch.
        release.set()
        results = await asyncio.gather(*tasks)

    # Only one network call was made.
    assert call_count == 1, f"Expected 1 Yahoo call, got {call_count}"

    # All callers received the same Ticker object.
    for r in results:
        assert r is ticker_obj


@pytest.mark.asyncio
@patch("yfmcp.yf_runner.asyncio.to_thread")
async def test_cache_hit_skips_yahoo_entirely(mock_to_thread: AsyncMock) -> None:
    """Second call (after TTL cache is warm) does not touch Yahoo at all."""
    ticker_obj = _make_ticker_mock("AAPL")
    call_count = 0

    async def counting_to_thread(func, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return func(*args, **kwargs)

    mock_to_thread.side_effect = counting_to_thread

    with patch("yfmcp.yf_runner.yf.Ticker", return_value=ticker_obj):
        first = await _get_ticker("AAPL")
        second = await _get_ticker("AAPL")

    assert call_count == 1
    assert first is ticker_obj
    assert second is ticker_obj


@pytest.mark.asyncio
@patch("yfmcp.yf_runner.asyncio.to_thread")
async def test_inflight_registry_cleaned_up_after_success(mock_to_thread: AsyncMock) -> None:
    """The ``_ticker_inflight`` dict must be empty after a successful fetch."""
    ticker_obj = _make_ticker_mock("TSLA")

    async def instant_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    mock_to_thread.side_effect = instant_to_thread

    with patch("yfmcp.yf_runner.yf.Ticker", return_value=ticker_obj):
        await _get_ticker("TSLA")

    assert "TSLA" not in _ticker_inflight


@pytest.mark.asyncio
@patch("yfmcp.yf_runner.asyncio.to_thread")
async def test_inflight_registry_cleaned_up_after_error(mock_to_thread: AsyncMock) -> None:
    """The ``_ticker_inflight`` dict must be empty even when the fetch raises."""
    mock_to_thread.side_effect = RuntimeError("network down")

    with patch("yfmcp.yf_runner.yf.Ticker"), pytest.raises(RuntimeError, match="network down"):
        await _get_ticker("BADSYM")

    assert "BADSYM" not in _ticker_inflight


@pytest.mark.asyncio
@patch("yfmcp.yf_runner.asyncio.to_thread")
async def test_waiters_receive_exception_when_fetch_fails(mock_to_thread: AsyncMock) -> None:
    """If the designated fetcher fails, all waiting coroutines should also raise."""
    release = asyncio.Event()
    call_count = 0

    async def failing_to_thread(func, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        await release.wait()
        raise ConnectionError("yahoo down")

    mock_to_thread.side_effect = failing_to_thread

    with patch("yfmcp.yf_runner.yf.Ticker"):
        tasks = [asyncio.create_task(_get_ticker("ERR")) for _ in range(3)]
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release.set()

        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Only one actual call to Yahoo.
    assert call_count == 1

    # The designated fetcher raises directly; the two waiters get it via the Future.
    errors = [r for r in results if isinstance(r, Exception)]
    assert len(errors) == 3
    for e in errors:
        assert isinstance(e, (ConnectionError, Exception))
