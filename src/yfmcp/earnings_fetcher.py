"""Batch earnings fetcher (yfinance_get_earnings tool backend).

Analyst estimates and earnings dates update infrequently within a trading day,
so a 1-hour TTL strikes the right balance between freshness and rate-limit safety.
"""

from __future__ import annotations

from typing import Any

import yfinance as yf
from loguru import logger

from yfmcp.batch import BatchProcessor
from yfmcp.batch import TtlCache
from yfmcp.yf_runner import _RETRYABLE_YFINANCE_EXCEPTIONS
from yfmcp.yf_runner import _is_rate_limit_error
from yfmcp.yf_runner import _run_yf

_CACHE_TTL_SECONDS = 60 * 60  # 1 hour

_cache = TtlCache(ttl_seconds=_CACHE_TTL_SECONDS)


async def _fetch_earnings(  # noqa: C901
    symbol: str,
    *,
    history_limit: int = 12,
) -> dict[str, Any]:
    """Fetch earnings beat/miss history, forward estimates, and EPS revision data."""
    try:
        ticker = await _run_yf(yf.Ticker, symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        if _is_rate_limit_error(exc):
            return {"error": "Rate limit reached — try again later"}
        return {"error": f"Temporary network error: {exc}"}
    except Exception as exc:
        return {"error": f"Failed to fetch earnings for '{symbol}': {exc}"}

    data: dict[str, Any] = {}

    try:
        dates_df = await _run_yf(ticker.get_earnings_dates, history_limit)
        if dates_df is not None and not dates_df.empty:
            dates_df = dates_df.copy()
            dates_df.index = dates_df.index.strftime("%Y-%m-%d %H:%M %Z")
            data["earnings_dates"] = dates_df.to_dict(orient="index")
    except Exception as exc:
        logger.warning("Failed to fetch earnings_dates for {}: {}", symbol, exc)

    try:
        ee = await _run_yf(ticker.get_earnings_estimate, True)
        if ee:
            data["earnings_estimate"] = ee
    except Exception as exc:
        logger.warning("Failed to fetch earnings_estimate for {}: {}", symbol, exc)

    try:
        re = await _run_yf(ticker.get_revenue_estimate, True)
        if re:
            data["revenue_estimate"] = re
    except Exception as exc:
        logger.warning("Failed to fetch revenue_estimate for {}: {}", symbol, exc)

    try:
        et = await _run_yf(ticker.get_eps_trend, True)
        if et:
            data["eps_trend"] = et
    except Exception as exc:
        logger.warning("Failed to fetch eps_trend for {}: {}", symbol, exc)

    try:
        er = await _run_yf(ticker.get_eps_revisions, True)
        if er:
            data["eps_revisions"] = er
    except Exception as exc:
        logger.warning("Failed to fetch eps_revisions for {}: {}", symbol, exc)

    if not data:
        return {"error": f"No earnings data available for '{symbol}'."}

    return {
        "data": data,
        "meta": {
            "historyLimit": history_limit,
            "warnings": [],
        },
    }


processor = BatchProcessor(
    fetch_fn=_fetch_earnings,
    cache=_cache,
    batch_size=5,
    batch_delay_seconds=0.3,
)
