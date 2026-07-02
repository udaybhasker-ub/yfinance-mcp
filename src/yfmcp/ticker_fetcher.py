"""Batch ticker-info fetcher (yfinance_get_ticker_info_batch tool backend).

Company fundamentals/profile data changes slowly intraday, so a 15-minute
in-memory cache keeps repeated portfolio-wide lookups cheap without serving
badly stale prices.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from yfmcp.batch import BatchProcessor
from yfmcp.batch import TtlCache
from yfmcp.yf_runner import _RETRYABLE_YFINANCE_EXCEPTIONS
from yfmcp.yf_runner import _get_ticker
from yfmcp.yf_runner import _is_rate_limit_error
from yfmcp.yf_runner import _run_yf

_CACHE_TTL_SECONDS = 15 * 60  # 15 minutes

_cache = TtlCache(ttl_seconds=_CACHE_TTL_SECONDS)


def _humanize_timestamps(info: dict[str, Any]) -> dict[str, Any]:
    for key, value in list(info.items()):
        if not isinstance(key, str) or not isinstance(value, int | float):
            continue
        if key.lower().endswith(("date", "start", "end", "timestamp", "time", "quarter")):
            try:
                info[key] = datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
    return info


async def _fetch_ticker_info(symbol: str) -> dict[str, Any]:
    """Fetch comprehensive ticker info for one symbol."""
    try:
        ticker = await _get_ticker(symbol)
        info = await _run_yf(lambda: ticker.info)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        if _is_rate_limit_error(exc):
            return {"error": "Rate limit reached — try again later"}
        return {"error": f"Temporary network error fetching ticker info for '{symbol}': {exc}"}
    except Exception as exc:
        return {"error": f"Failed to fetch ticker info for '{symbol}': {exc}"}

    if not info:
        return {"error": f"No information available for symbol '{symbol}'. The symbol may be invalid or delisted."}

    return {
        "data": _humanize_timestamps(info),
        "meta": {"warnings": []},
    }


processor = BatchProcessor(
    fetch_fn=_fetch_ticker_info,
    cache=_cache,
    batch_size=5,
    batch_delay_seconds=0.3,
)
