"""Shared yfinance async runner utilities.

Provides the concurrency-controlled ``_run_yf`` coroutine and related helpers
used by both ``server.py`` (all MCP tools) and ``quote_fetcher.py``.

yfinance's ``YfData`` is a process-wide singleton: every Ticker/Search/Sector
call shares one curl_cffi session and cookie/crumb lock across all threads.
Firing many concurrent calls at it causes them to serialize on that shared
session and choke each other.  ``_run_yf`` gates every yfinance call behind a
semaphore and bounds each with a timeout so a stalled call fails fast instead
of hanging the MCP client.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger
from yfinance.exceptions import YFRateLimitError

from yfmcp.utils import create_error_response

_RETRYABLE_YFINANCE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
    YFRateLimitError,
)

_YF_CALL_CONCURRENCY = asyncio.Semaphore(4)
_YF_CALL_TIMEOUT_SECONDS = 30.0


def _describe_yf_call(func, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    name = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or repr(func)
    parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    return f"{name}({', '.join(parts)})" if parts else name


async def _run_yf(func, *args, **kwargs):
    call_desc = _describe_yf_call(func, args, kwargs)
    async with _YF_CALL_CONCURRENCY:
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=_YF_CALL_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.error(
                "yahoo_call={} status=failed duration={:.2f}s error={}: {}",
                call_desc,
                time.monotonic() - start,
                type(exc).__name__,
                exc,
            )
            raise
        else:
            logger.info(
                "yahoo_call={} status=success duration={:.2f}s",
                call_desc,
                time.monotonic() - start,
            )
            return result


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, YFRateLimitError)


def _is_retryable_yfinance_error(exc: BaseException) -> bool:
    return isinstance(exc, _RETRYABLE_YFINANCE_EXCEPTIONS)


def _select_retryable_exception(exceptions: list[Exception]) -> BaseException:
    rate_limit_exception = next((exc for exc in exceptions if _is_rate_limit_error(exc)), None)
    return rate_limit_exception or exceptions[0]


def _create_retryable_error_response(action: str, exc: BaseException, details: dict[str, Any]) -> str:
    if _is_rate_limit_error(exc):
        message = f"Rate limit reached while {action}. Try again later."
    else:
        message = f"Temporary network issue while {action}. Try again later."
    return create_error_response(message, error_code="NETWORK_ERROR", details={**details, "exception": str(exc)})
