"""Batch news fetcher (yfinance_get_ticker_news tool backend).

News is time-sensitive — a 2-minute TTL keeps burst calls cheap without
serving meaningfully stale headlines.
"""

from __future__ import annotations

from typing import Any

from yfmcp.batch import BatchProcessor
from yfmcp.batch import TtlCache
from yfmcp.yf_runner import _RETRYABLE_YFINANCE_EXCEPTIONS
from yfmcp.yf_runner import _get_ticker
from yfmcp.yf_runner import _is_rate_limit_error
from yfmcp.yf_runner import _run_yf

_CACHE_TTL_SECONDS = 2 * 60  # 2 minutes

_cache = TtlCache(ttl_seconds=_CACHE_TTL_SECONDS)


async def _fetch_news(symbol: str) -> dict[str, Any]:
    """Fetch recent news articles for one ticker."""
    try:
        ticker = await _get_ticker(symbol)
        news = await _run_yf(ticker.get_news)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        if _is_rate_limit_error(exc):
            return {"error": "Rate limit reached — try again later"}
        return {"error": f"Temporary network error: {exc}"}
    except Exception as exc:
        return {"error": f"Failed to fetch news for '{symbol}': {exc}"}

    if not news:
        return {  # noqa: E501
            "error": f"No news articles available for '{symbol}'. The symbol may be invalid or have no recent coverage."
        }

    return {
        "data": news,
        "meta": {
            "articleCount": len(news),
            "warnings": [],
        },
    }


processor = BatchProcessor(
    fetch_fn=_fetch_news,
    cache=_cache,
    batch_size=5,
    batch_delay_seconds=0.2,
)
