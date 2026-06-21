"""Shared yfinance async runner utilities.

Provides the concurrency-controlled ``_run_yf`` coroutine and related helpers
used by both ``server.py`` (all MCP tools) and ``quote_fetcher.py``.

yfinance's ``YfData`` is a process-wide singleton: every Ticker/Search/Sector
call shares one curl_cffi session and cookie/crumb lock across all threads.
Firing many concurrent calls at it causes them to serialize on that shared
session and choke each other.  ``_run_yf`` gates every yfinance call behind a
semaphore and bounds each with a timeout so a stalled call fails fast instead
of hanging the MCP client.

``_get_ticker`` implements single-flight coalescing: when multiple tool calls
fire for the same symbol simultaneously (e.g. get_quote + get_analyst +
get_earnings all triggered by one AI skill within milliseconds of each other),
only the first coroutine actually contacts Yahoo Finance.  The remaining
coroutines await an ``asyncio.Future`` that the first coroutine resolves once
its fetch completes.  Subsequent callers within the 60-second TTL window are
served from the ``TtlCache`` without any network call.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import yfinance as yf
from loguru import logger
from yfinance.exceptions import YFRateLimitError

from yfmcp.batch import TtlCache
from yfmcp.utils import create_error_response

_RETRYABLE_YFINANCE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
    YFRateLimitError,
)

_YF_CALL_CONCURRENCY = asyncio.Semaphore(4)
_YF_CALL_TIMEOUT_SECONDS = 30.0

# 60-second Ticker object cache — keyed by uppercase symbol.
_TICKER_CACHE_TTL_SECONDS = 60
_ticker_cache = TtlCache(ttl_seconds=_TICKER_CACHE_TTL_SECONDS)

# Single-flight registry: symbol → in-flight Future.  Callers that arrive while
# a fetch is already running await the same Future rather than firing a duplicate
# request to Yahoo Finance.  Access is serialised by ``_ticker_inflight_lock``.
_ticker_inflight: dict[str, asyncio.Future] = {}
_ticker_inflight_lock = asyncio.Lock()


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


async def _get_ticker(symbol: str) -> Any:
    """Return a ``yf.Ticker`` for *symbol* with single-flight coalescing.

    Execution order for concurrent callers targeting the same symbol:

    1. **Cache hit** — served immediately, no lock, no Yahoo call.
    2. **First miss** — acquires ``_ticker_inflight_lock`` briefly, creates a
       ``Future``, registers it, releases the lock, then fetches from Yahoo.
       On completion the Future is resolved and the result is written to
       ``_ticker_cache``.
    3. **Subsequent misses while fetch is in-flight** — acquire the lock, find
       the existing Future, release the lock, then ``await`` it.  They share
       the result of the single in-progress fetch rather than each firing their
       own Yahoo request.

    This collapses e.g. four simultaneous ``get_quote / get_analyst /
    get_earnings / get_financials`` calls for the same ticker into one network
    round-trip.
    """
    key = symbol.strip().upper()

    # Fast path — serve from TTL cache without touching the lock.
    cached = await _ticker_cache.get(key, {})
    if cached is not None:
        value, _ = cached
        return value

    # Slow path — check for an in-flight fetch or register ourselves as the
    # designated fetcher.  Hold the lock only long enough to inspect/update the
    # dict (no I/O inside the lock).
    own_fetch = False
    async with _ticker_inflight_lock:
        if key in _ticker_inflight:
            fut: asyncio.Future = _ticker_inflight[key]
        else:
            fut = asyncio.get_running_loop().create_future()
            _ticker_inflight[key] = fut
            own_fetch = True

    if not own_fetch:
        # Wait for the designated fetcher and return its result (or re-raise).
        return await fut

    # We are the designated fetcher.
    try:
        ticker = await _run_yf(yf.Ticker, key)
        await _ticker_cache.set(key, {}, ticker)
        fut.set_result(ticker)
        return ticker
    except Exception as exc:
        if not fut.done():
            fut.set_exception(exc)
        raise
    finally:
        async with _ticker_inflight_lock:
            _ticker_inflight.pop(key, None)


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
