"""QuoteFetcher — batch yfinance quote fetching with a curated, analysis-ready field set.

Mirrors the structure of FinMCP's QuoteTools class: all field definitions, timestamp
handling, and per-ticker/batch fetch logic live here.  The MCP tool function in
server.py is a thin wrapper that calls ``QuoteFetcher.fetch_batch``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import datetime
from typing import Any

import yfinance as yf

from yfmcp.yf_runner import _RETRYABLE_YFINANCE_EXCEPTIONS
from yfmcp.yf_runner import _is_rate_limit_error
from yfmcp.yf_runner import _run_yf


class QuoteFetcher:
    # Curated fields returned by default.  Much smaller than the full ~127-field
    # ticker.info blob, so parallel multi-ticker calls stay well within proxy limits.
    DEFAULT_FIELDS: tuple[str, ...] = (
        # Price
        "currentPrice",
        "regularMarketChange",
        "regularMarketChangePercent",
        "regularMarketPreviousClose",
        "regularMarketOpen",
        "regularMarketVolume",
        "averageVolume",
        "averageVolume10days",
        # Valuation
        "marketCap",
        "trailingPE",
        "forwardPE",
        "trailingEps",
        "forwardEps",
        "pegRatio",
        "priceToBook",
        # Dividends
        "dividendRate",
        "dividendYield",
        "exDividendDate",
        # Risk / technicals
        "beta",
        "fiftyDayAverage",
        "twoHundredDayAverage",
        # Shares
        "sharesOutstanding",
        # Analyst
        "targetMeanPrice",
        "recommendationKey",
        "numberOfAnalystOpinions",
        # Earnings
        "earningsTimestamp",
        # Growth
        "revenueGrowth",
        "earningsGrowth",
        "profitMargins",
        # Identity
        "longName",
        "sector",
        "industry",
    )

    # Fields whose numeric values are Unix timestamps → convert to YYYY-MM-DD strings.
    TIMESTAMP_FIELDS: frozenset[str] = frozenset(
        {
            "exDividendDate",
            "earningsTimestamp",
            "dividendDate",
            "lastDividendDate",
            "lastFiscalYearEnd",
            "nextFiscalYearEnd",
            "mostRecentQuarter",
        }
    )

    BATCH_SIZE: int = 10
    BATCH_DELAY_SECONDS: float = 0.2

    # Thresholds for data-staleness warnings (milliseconds).
    _FIFTEEN_MIN_MS: int = 15 * 60 * 1000
    _FIVE_MIN_MS: int = 5 * 60 * 1000

    @classmethod
    def _extract_fields(cls, info: dict[str, Any], fields: list[str] | None) -> dict[str, Any]:
        """Build the curated data dict (selected fields + range objects) from raw ticker.info."""
        field_set = fields if fields else cls.DEFAULT_FIELDS
        data: dict[str, Any] = {}
        for field in field_set:
            if field not in info:
                continue
            value = info[field]
            if isinstance(value, int | float) and field in cls.TIMESTAMP_FIELDS:
                with contextlib.suppress(Exception):
                    value = datetime.fromtimestamp(value).strftime("%Y-%m-%d")
            data[field] = value

        # Always include structured range objects (FinMCP convention).
        day_low = info.get("regularMarketDayLow") or info.get("dayLow")
        day_high = info.get("regularMarketDayHigh") or info.get("dayHigh")
        if day_low is not None and day_high is not None:
            data["regularMarketDayRange"] = {"low": day_low, "high": day_high}

        wk52_low = info.get("fiftyTwoWeekLow")
        wk52_high = info.get("fiftyTwoWeekHigh")
        if wk52_low is not None and wk52_high is not None:
            data["fiftyTwoWeekRange"] = {"low": wk52_low, "high": wk52_high}

        return data

    @classmethod
    async def fetch_single(cls, symbol: str, fields: list[str] | None) -> dict[str, Any]:
        """Fetch ticker.info for one symbol and return a FinMCP-shaped result dict."""
        try:
            ticker = await _run_yf(yf.Ticker, symbol)
            info = await _run_yf(lambda: ticker.info)
        except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
            if _is_rate_limit_error(exc):
                msg = "Rate limit reached — try again later"
            else:
                msg = f"Temporary network error: {exc}"
            return {"error": msg}
        except Exception as exc:
            return {"error": str(exc)}

        if not info:
            return {"error": f"No data for '{symbol}' — symbol may be invalid or delisted"}

        # Capture data age (ms) before any timestamp mutation.
        raw_market_time = info.get("regularMarketTime")
        data_age_ms: int = 0
        if isinstance(raw_market_time, int | float) and raw_market_time > 0:
            data_age_ms = max(0, int(time.time() * 1000 - raw_market_time * 1000))

        data = cls._extract_fields(info, fields)

        non_null = sum(1 for v in data.values() if v is not None)
        completeness = round(non_null / len(data), 3) if data else 0.0

        warnings: list[str] = []
        if data_age_ms > cls._FIFTEEN_MIN_MS:
            warnings.append("Data is delayed (15+ minutes)")
        elif data_age_ms > cls._FIVE_MIN_MS:
            warnings.append("Data may not be real-time (5+ minutes old)")

        return {
            "data": data,
            "meta": {
                "dataAge": data_age_ms,
                "completenessScore": completeness,
                "warnings": warnings,
            },
        }

    @classmethod
    async def fetch_batch(cls, symbols: list[str], fields: list[str] | None) -> dict[str, Any]:
        """Fetch quotes for multiple symbols in batches, returning a FinMCP-shaped envelope."""
        results: dict[str, Any] = {}
        errors: list[dict[str, str]] = []

        normalised = [s.strip().upper() for s in symbols]
        batches = [normalised[i : i + cls.BATCH_SIZE] for i in range(0, len(normalised), cls.BATCH_SIZE)]

        for batch_idx, batch in enumerate(batches):
            for symbol in batch:
                result = await cls.fetch_single(symbol, fields)
                if "error" in result:
                    errors.append({"symbol": symbol, "error": result["error"]})
                else:
                    results[symbol] = result

            if batch_idx < len(batches) - 1:
                await asyncio.sleep(cls.BATCH_DELAY_SECONDS)

        return {
            "results": results,
            "summary": {
                "totalRequested": len(normalised),
                "totalReturned": len(results),
                "errors": errors,
            },
        }
