"""Batch financials fetcher (yfinance_get_financials tool backend).

Financials update once per quarter, so a 6-hour in-memory cache is aggressive
enough to survive burst usage without serving meaningfully stale data.

``_build_financials_response`` was previously inlined in server.py and is now
co-located here where it belongs.
"""

from __future__ import annotations

from typing import Any

from yfmcp.batch import BatchProcessor
from yfmcp.batch import TtlCache
from yfmcp.yf_runner import _RETRYABLE_YFINANCE_EXCEPTIONS
from yfmcp.yf_runner import _get_ticker
from yfmcp.yf_runner import _is_rate_limit_error
from yfmcp.yf_runner import _run_yf

_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours

_cache = TtlCache(ttl_seconds=_CACHE_TTL_SECONDS)

_INCOME_FIELDS = [
    "EBIT",
    "Net Income",
    "Tax Provision",
    "Pretax Income",
    "Interest Expense",
    "Total Revenue",
    "Operating Income",
    "EBITDA",
    "Normalized Income",
]

_BALANCE_FIELDS = [
    "Stockholders Equity",
    "Total Debt",
    "Cash And Cash Equivalents",
    "Invested Capital",
    "Net Debt",
    "Total Assets",
    "Total Liabilities Net Minority Interest",
    "Net Tangible Assets",
    "Tangible Book Value",
]

_CASH_FLOW_FIELDS = [
    "Operating Cash Flow",
    "Free Cash Flow",
    "Capital Expenditure",
    "Net Income From Continuing Operations",
    "Depreciation And Amortization",
    "Change In Working Capital",
    "Cash Dividends Paid",
]


def _build_financials_response(income_stmt: Any, balance_sheet: Any, cash_flow: Any) -> dict[str, Any]:
    """Build a financials dict from yfinance DataFrames."""
    result: dict[str, Any] = {}

    if income_stmt is not None and not income_stmt.empty:
        available = [f for f in _INCOME_FIELDS if f in income_stmt.index]
        result["income_statement"] = {
            field: {str(col.date()): income_stmt.loc[field, col] for col in income_stmt.columns}
            for field in available
        }

    if balance_sheet is not None and not balance_sheet.empty:
        available = [f for f in _BALANCE_FIELDS if f in balance_sheet.index]
        result["balance_sheet"] = {
            field: {str(col.date()): balance_sheet.loc[field, col] for col in balance_sheet.columns}
            for field in available
        }

    if cash_flow is not None and not cash_flow.empty:
        available = [f for f in _CASH_FLOW_FIELDS if f in cash_flow.index]
        result["cash_flow"] = {
            field: {str(col.date()): cash_flow.loc[field, col] for col in cash_flow.columns}
            for field in available
        }

    return result


async def _fetch_financials(  # noqa: C901
    symbol: str,
    *,
    frequency: str = "annual",
) -> dict[str, Any]:
    """Fetch income statement, balance sheet, and cash flow for one ticker."""
    if frequency not in {"annual", "quarterly", "ttm"}:
        return {"error": f"Invalid frequency '{frequency}'. Valid options: 'annual', 'quarterly', 'ttm'."}

    try:
        ticker = await _get_ticker(symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        if _is_rate_limit_error(exc):
            return {"error": "Rate limit reached — try again later"}
        return {"error": f"Temporary network error: {exc}"}
    except Exception as exc:
        return {"error": f"Failed to fetch financials for '{symbol}': {exc}"}

    income_stmt = balance_sheet = cash_flow = None
    try:
        if frequency == "annual":
            income_stmt = await _run_yf(lambda: ticker.income_stmt)
            balance_sheet = await _run_yf(lambda: ticker.balance_sheet)
            cash_flow = await _run_yf(lambda: ticker.cashflow)
        elif frequency == "quarterly":
            income_stmt = await _run_yf(lambda: ticker.quarterly_income_stmt)
            balance_sheet = await _run_yf(lambda: ticker.quarterly_balance_sheet)
            cash_flow = await _run_yf(lambda: ticker.quarterly_cashflow)
        else:  # ttm
            income_stmt = await _run_yf(lambda: ticker.ttm_income_stmt)
            # TTM balance sheet and cash flow are not directly available in yfinance.
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        if _is_rate_limit_error(exc):
            return {"error": "Rate limit reached — try again later"}
        return {"error": f"Temporary network error fetching financials for '{symbol}': {exc}"}
    except Exception as exc:
        return {"error": f"Failed to fetch financials for '{symbol}': {exc}"}

    data = _build_financials_response(income_stmt, balance_sheet, cash_flow)
    if not data:
        return {"error": f"No financial data available for '{symbol}' with frequency='{frequency}'."}

    return {
        "data": data,
        "meta": {
            "frequency": frequency,
            "warnings": [],
        },
    }


processor = BatchProcessor(
    fetch_fn=_fetch_financials,
    cache=_cache,
    batch_size=5,
    batch_delay_seconds=0.3,
)
