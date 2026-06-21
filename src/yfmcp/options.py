"""Options service helpers — per-date fetch logic and error builders for option chain tools."""

from __future__ import annotations

from typing import Any

import yfinance as yf

from yfmcp.types import OptionChainType
from yfmcp.utils import create_error_response
from yfmcp.yf_runner import _create_retryable_error_response
from yfmcp.yf_runner import _is_retryable_yfinance_error
from yfmcp.yf_runner import _run_yf
from yfmcp.yf_runner import _select_retryable_exception


async def _fetch_option_chain_for_date(
    ticker: yf.Ticker,
    date: str,
    option_type: OptionChainType,
) -> dict[str, Any]:
    """Fetch option chain for a single expiration date."""
    opt = await _run_yf(lambda d=date: ticker.option_chain(d))

    calls_df = opt.calls
    puts_df = opt.puts
    date_data: dict[str, Any] = {}

    if calls_df is not None and not calls_df.empty and option_type in {"all", "calls"}:
        calls_df = calls_df.copy()
        calls_df["optionType"] = "CALL"
        date_data["calls"] = calls_df.to_dict(orient="records")

    if puts_df is not None and not puts_df.empty and option_type in {"all", "puts"}:
        puts_df = puts_df.copy()
        puts_df["optionType"] = "PUT"
        date_data["puts"] = puts_df.to_dict(orient="records")

    return {date: date_data} if date_data else {}


def _create_option_dates_fetch_error(symbol: str, exc: Exception, api_message: str) -> str:
    if _is_retryable_yfinance_error(exc):
        return _create_retryable_error_response(f"fetching option dates for '{symbol}'", exc, {"symbol": symbol})
    return create_error_response(
        api_message,
        error_code="API_ERROR",
        details={"symbol": symbol, "exception": str(exc)},
    )


def _create_option_chain_fetch_error(
    symbol: str,
    dates_to_fetch: list[str],
    fetch_errors: list[tuple[str, Exception]],
) -> str:
    failed_dates = [date for date, _ in fetch_errors]

    if len(dates_to_fetch) == 1:
        failed_date, exc = fetch_errors[0]
        if _is_retryable_yfinance_error(exc):
            return _create_retryable_error_response(
                f"fetching option chain for '{symbol}' on '{failed_date}'",
                exc,
                {"symbol": symbol, "expiration_date": failed_date},
            )
        return create_error_response(
            f"Failed to fetch option chain for '{symbol}' on '{failed_date}'.",
            error_code="API_ERROR",
            details={"symbol": symbol, "expiration_date": failed_date, "exception": str(exc)},
        )

    retryable_exceptions = [exc for _, exc in fetch_errors if _is_retryable_yfinance_error(exc)]

    if retryable_exceptions:
        return _create_retryable_error_response(
            f"fetching option chain for '{symbol}'",
            _select_retryable_exception(retryable_exceptions),
            {
                "symbol": symbol,
                "dates_requested": dates_to_fetch,
                "failed_dates": failed_dates,
            },
        )

    representative_exception = fetch_errors[0][1]
    return create_error_response(
        f"Failed to fetch option chain for '{symbol}' for all requested dates.",
        error_code="API_ERROR",
        details={
            "symbol": symbol,
            "dates_requested": dates_to_fetch,
            "failed_dates": failed_dates,
            "exception": str(representative_exception),
        },
    )
