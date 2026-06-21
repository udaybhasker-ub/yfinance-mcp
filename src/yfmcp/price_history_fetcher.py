"""Batch price-history fetcher (yfinance_get_price_history tool backend).

Exposes a module-level ``processor`` (``BatchProcessor``) that server.py uses
for multi-ticker calls.  Single-ticker chart rendering is intentionally kept in
server.py, since it returns an ``ImageContent`` rather than a JSON envelope.

Cache TTL: 15 minutes — balances live intraday freshness with rate-limit safety.
"""

from __future__ import annotations

from typing import Any

import yfinance as yf

from yfmcp.batch import BatchProcessor
from yfmcp.batch import TtlCache
from yfmcp.yf_runner import _RETRYABLE_YFINANCE_EXCEPTIONS
from yfmcp.yf_runner import _is_rate_limit_error
from yfmcp.yf_runner import _run_yf

_CACHE_TTL_SECONDS = 15 * 60  # 15 min

_cache = TtlCache(ttl_seconds=_CACHE_TTL_SECONDS)


async def _fetch_price_history(
    symbol: str,
    *,
    period: str = "1mo",
    interval: str = "1d",
    prepost: bool = False,
) -> dict[str, Any]:
    """Fetch OHLCV history for one ticker; return BatchProcessor-compatible dict."""
    try:
        ticker = await _run_yf(yf.Ticker, symbol)
        df = await _run_yf(
            ticker.history,
            period=period,
            interval=interval,
            prepost=prepost,
            rounding=True,
        )
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        msg = "Rate limit reached — try again later" if _is_rate_limit_error(exc) else f"Temporary network error: {exc}"
        return {"error": msg}
    except Exception as exc:
        return {"error": f"Failed to fetch price history for '{symbol}': {exc}"}

    if df.empty:
        return {
            "error": (
                f"No price data for '{symbol}' with period='{period}' interval='{interval}'. "
                "Check the symbol, or try a longer period / daily interval."
            )
        }

    return {
        "data": df.to_markdown(),
        "meta": {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "prepost": prepost,
            "rows": len(df),
            "warnings": [],
        },
    }


processor = BatchProcessor(
    fetch_fn=_fetch_price_history,
    cache=_cache,
    batch_size=5,
    batch_delay_seconds=0.3,
)
