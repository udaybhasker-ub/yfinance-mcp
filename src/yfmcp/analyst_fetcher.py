"""Batch analyst fetcher (yfinance_get_analyst tool backend).

Consensus price targets and firm-level upgrade/downgrade history update
infrequently, so a 1-hour TTL is appropriate.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from yfmcp.batch import BatchProcessor
from yfmcp.batch import TtlCache
from yfmcp.yf_runner import _RETRYABLE_YFINANCE_EXCEPTIONS
from yfmcp.yf_runner import _get_ticker
from yfmcp.yf_runner import _is_rate_limit_error
from yfmcp.yf_runner import _run_yf

_CACHE_TTL_SECONDS = 60 * 60  # 1 hour

_cache = TtlCache(ttl_seconds=_CACHE_TTL_SECONDS)


async def _fetch_analyst(  # noqa: C901
    symbol: str,
    *,
    upgrades_limit: int = 20,
) -> dict[str, Any]:
    """Fetch analyst consensus, price targets, and upgrade/downgrade history."""
    try:
        ticker = await _get_ticker(symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        if _is_rate_limit_error(exc):
            return {"error": "Rate limit reached — try again later"}
        return {"error": f"Temporary network error: {exc}"}
    except Exception as exc:
        return {"error": f"Failed to fetch analyst data for '{symbol}': {exc}"}

    data: dict[str, Any] = {}

    try:
        pt = await _run_yf(ticker.get_analyst_price_targets)
        if pt:
            data["price_targets"] = pt
    except Exception as exc:
        logger.warning("Failed to fetch analyst_price_targets for {}: {}", symbol, exc)

    try:
        rec_df = await _run_yf(ticker.get_recommendations, False)
        if rec_df is not None and not rec_df.empty:
            data["recommendations"] = rec_df.to_dict(orient="records")
    except Exception as exc:
        logger.warning("Failed to fetch recommendations for {}: {}", symbol, exc)

    try:
        ud_df = await _run_yf(ticker.get_upgrades_downgrades, False)
        if ud_df is not None and not ud_df.empty:
            ud_df = ud_df.copy().head(upgrades_limit)
            ud_df.index = ud_df.index.strftime("%Y-%m-%d")
            data["upgrades_downgrades"] = ud_df.reset_index().to_dict(orient="records")
    except Exception as exc:
        logger.warning("Failed to fetch upgrades_downgrades for {}: {}", symbol, exc)

    if not data:
        return {"error": f"No analyst data available for '{symbol}'."}

    return {
        "data": data,
        "meta": {
            "upgradesLimit": upgrades_limit,
            "warnings": [],
        },
    }


processor = BatchProcessor(
    fetch_fn=_fetch_analyst,
    cache=_cache,
    batch_size=5,
    batch_delay_seconds=0.3,
)
