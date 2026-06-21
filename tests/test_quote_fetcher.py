"""Unit tests for QuoteFetcher (yfinance_get_quote tool backend)."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from yfinance.exceptions import YFRateLimitError

from yfmcp.quote_fetcher import QuoteFetcher
from yfmcp.quote_fetcher import _quote_cache
from yfmcp.yf_runner import _ticker_cache
from yfmcp.yf_runner import _ticker_inflight


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    """Clear quote and ticker caches between tests to prevent cross-test contamination."""
    _quote_cache.clear_sync()
    _ticker_cache.clear_sync()
    _ticker_inflight.clear()


async def _run_to_thread(func, *args, **kwargs):
    if callable(func):
        return func(*args, **kwargs)
    return func


@pytest.mark.asyncio
@patch("yfmcp.quote_fetcher._get_ticker", new_callable=AsyncMock)
@patch("yfmcp.yf_runner.asyncio.to_thread")
async def test_fetch_single_returns_curated_default_fields(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.info = {
        "currentPrice": 150.0,
        "marketCap": 2_000_000_000,
        "longName": "Apple Inc.",
        "regularMarketDayLow": 148.0,
        "regularMarketDayHigh": 151.0,
        "fiftyTwoWeekLow": 120.0,
        "fiftyTwoWeekHigh": 199.0,
        "regularMarketTime": None,
        "irrelevantField": "ignored",
    }
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await QuoteFetcher.fetch_single("AAPL", None)

    assert result["data"]["currentPrice"] == 150.0
    assert result["data"]["longName"] == "Apple Inc."
    assert "irrelevantField" not in result["data"]
    assert result["data"]["regularMarketDayRange"] == {"low": 148.0, "high": 151.0}
    assert result["data"]["fiftyTwoWeekRange"] == {"low": 120.0, "high": 199.0}
    assert result["meta"]["dataAge"] == 0
    assert result["meta"]["warnings"] == []


@pytest.mark.asyncio
@patch("yfmcp.quote_fetcher._get_ticker", new_callable=AsyncMock)
@patch("yfmcp.yf_runner.asyncio.to_thread")
async def test_fetch_single_respects_explicit_fields(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.info = {"currentPrice": 10.0, "trailingPE": 5.0, "sector": "Tech"}
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await QuoteFetcher.fetch_single("NVDA", ["currentPrice"])

    assert set(result["data"].keys()) == {"currentPrice"}


@pytest.mark.asyncio
@patch("yfmcp.quote_fetcher._get_ticker", new_callable=AsyncMock)
@patch("yfmcp.yf_runner.asyncio.to_thread")
async def test_fetch_single_converts_timestamp_fields(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.info = {"exDividendDate": 1700000000}
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await QuoteFetcher.fetch_single("AAPL", ["exDividendDate"])

    assert result["data"]["exDividendDate"] == "2023-11-14"


@pytest.mark.asyncio
@patch("yfmcp.quote_fetcher._get_ticker", new_callable=AsyncMock)
@patch("yfmcp.yf_runner.asyncio.to_thread")
async def test_fetch_single_no_data_returns_error(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.info = {}
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await QuoteFetcher.fetch_single("BADSYM", None)

    assert "error" in result
    assert "BADSYM" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "expected_substring"),
    [
        (TimeoutError("timed out"), "Temporary network error"),
        (YFRateLimitError(), "Rate limit reached"),
    ],
)
@patch("yfmcp.quote_fetcher._get_ticker", new_callable=AsyncMock)
@patch("yfmcp.yf_runner.asyncio.to_thread")
async def test_fetch_single_retryable_errors(
    mock_to_thread: AsyncMock, mock_ticker: MagicMock, exception: Exception, expected_substring: str
) -> None:
    mock_ticker.side_effect = exception
    mock_to_thread.side_effect = _run_to_thread

    result = await QuoteFetcher.fetch_single("AAPL", None)

    assert expected_substring in result["error"]


@pytest.mark.asyncio
@patch("yfmcp.quote_fetcher._get_ticker", new_callable=AsyncMock)
@patch("yfmcp.yf_runner.asyncio.to_thread")
async def test_fetch_single_unexpected_error_returns_message(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    mock_ticker.side_effect = RuntimeError("boom")
    mock_to_thread.side_effect = _run_to_thread

    result = await QuoteFetcher.fetch_single("AAPL", None)

    assert result["error"] == "boom"


@pytest.mark.asyncio
@patch("yfmcp.quote_fetcher.QuoteFetcher.fetch_single")
async def test_fetch_batch_separates_results_and_errors(mock_fetch_single: AsyncMock) -> None:
    async def fake_fetch_single(symbol, fields, *, no_cache=False):
        if symbol == "BAD":
            return {"error": "No data for 'BAD'"}
        return {"data": {"currentPrice": 1.0}, "meta": {"dataAge": 0, "completenessScore": 1.0, "warnings": []}}

    mock_fetch_single.side_effect = fake_fetch_single

    result = await QuoteFetcher.fetch_batch(["aapl", "BAD"], None)

    assert "AAPL" in result["results"]
    assert "BAD" not in result["results"]
    assert result["summary"]["totalRequested"] == 2
    assert result["summary"]["totalReturned"] == 1
    assert result["summary"]["errors"] == [{"symbol": "BAD", "error": "No data for 'BAD'"}]


@pytest.mark.asyncio
@patch("yfmcp.quote_fetcher.QuoteFetcher.fetch_single")
async def test_fetch_batch_batches_large_symbol_lists(mock_fetch_single: AsyncMock) -> None:
    mock_fetch_single.return_value = {"data": {}, "meta": {"dataAge": 0, "completenessScore": 0.0, "warnings": []}}
    symbols = [f"SYM{i}" for i in range(12)]

    result = await QuoteFetcher.fetch_batch(symbols, None)

    assert result["summary"]["totalRequested"] == 12
    assert result["summary"]["totalReturned"] == 12
    assert mock_fetch_single.call_count == 12
