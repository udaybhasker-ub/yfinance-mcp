"""Generic batch-processing framework for multi-ticker MCP tools.

Provides two building blocks:

- ``TtlCache`` — async-safe in-memory cache with per-entry TTL.
- ``BatchProcessor`` — wraps any async ``fetch_fn(symbol, **kwargs) -> dict``
  and adds batching, concurrency control, caching, and a standard FinMCP-shaped
  response envelope.

Usage::

    _cache = TtlCache(ttl_seconds=3600)
    _processor = BatchProcessor(fetch_fn=_fetch_financials, cache=_cache)

    result = await _processor.run(["AAPL", "NVDA"], frequency="annual")
    # -> {
    #      "results": {"AAPL": {"data": ..., "meta": {"fromCache": False, ...}}},
    #      "summary": {"totalRequested": 2, "totalReturned": 1, "errors": [...]}
    #    }

``fetch_fn`` contract
---------------------
Must accept ``symbol: str`` as the first positional argument, plus any keyword
arguments forwarded from ``BatchProcessor.run(**kwargs)``.

- On success → return ``{"data": <any>, "meta": {"warnings": [...], ...}}``
- On failure → return ``{"error": "<human-readable message>"}``

``BatchProcessor`` stamps ``meta.fromCache`` and ``meta.cacheAge`` (seconds)
before returning, so ``fetch_fn`` implementations do not need to handle caching.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any


class TtlCache:
    """Async-safe in-memory TTL cache.

    Stores ``(value, cached_at_monotonic)`` pairs.  Expired entries are evicted
    lazily on the next ``get`` for the same key.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        # key -> (value, cached_at)
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    def _make_key(self, symbol: str, kwargs: dict[str, Any]) -> str:
        """Stable cache key from symbol + serialised kwargs."""
        return json.dumps({"symbol": symbol, **kwargs}, sort_keys=True, default=str)

    async def get(self, symbol: str, kwargs: dict[str, Any]) -> tuple[Any, float] | None:
        """Return ``(value, cached_at)`` or ``None`` if missing/expired."""
        key = self._make_key(symbol, kwargs)
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, cached_at = entry
            if time.monotonic() - cached_at > self._ttl:
                del self._store[key]
                return None
            return value, cached_at

    async def set(self, symbol: str, kwargs: dict[str, Any], value: Any) -> None:
        """Store ``value`` keyed by ``symbol + kwargs``."""
        key = self._make_key(symbol, kwargs)
        async with self._lock:
            self._store[key] = (value, time.monotonic())

    async def clear(self) -> None:
        """Evict all entries (useful in tests)."""
        async with self._lock:
            self._store.clear()

    def clear_sync(self) -> None:
        """Evict all entries without acquiring the async lock.

        Safe to call from sync test fixtures before any async test body runs,
        because no coroutines are executing concurrently at that point.
        """
        self._store.clear()


FetchFn = Callable[..., Awaitable[dict[str, Any]]]


class BatchProcessor:
    """Run an async ``fetch_fn`` across many symbols with batching and caching.

    Parameters
    ----------
    fetch_fn:
        ``async def f(symbol: str, **kwargs) -> dict`` — see module docstring
        for the required return shape.
    cache:
        Optional ``TtlCache`` instance.  Pass ``None`` to disable caching
        (e.g., for highly time-sensitive data such as live news).
    batch_size:
        Symbols dispatched per batch before the inter-batch delay fires.
        Within a batch, all fetches run concurrently via ``asyncio.gather``;
        the global ``_YF_CALL_CONCURRENCY`` semaphore in ``yf_runner`` bounds
        actual parallelism to 4.
    batch_delay_seconds:
        Sleep duration between batches to respect Yahoo Finance rate limits.
    """

    def __init__(
        self,
        fetch_fn: FetchFn,
        cache: TtlCache | None = None,
        batch_size: int = 5,
        batch_delay_seconds: float = 0.3,
    ) -> None:
        self._fetch_fn = fetch_fn
        self._cache = cache
        self._batch_size = batch_size
        self._batch_delay = batch_delay_seconds

    async def _fetch_one(self, symbol: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Check cache then live-fetch; stamps ``meta.fromCache`` / ``meta.cacheAge``."""
        if self._cache is not None:
            cached = await self._cache.get(symbol, kwargs)
            if cached is not None:
                value, cached_at = cached
                age_seconds = int(time.monotonic() - cached_at)
                # Deep-copy only the layers we mutate to avoid aliasing.
                result: dict[str, Any] = {**value, "meta": {**value.get("meta", {}), "fromCache": True, "cacheAge": age_seconds}}  # noqa: E501
                return result

        result = await self._fetch_fn(symbol, **kwargs)

        if "error" not in result:
            if self._cache is not None:
                await self._cache.set(symbol, kwargs, result)
            meta = result.setdefault("meta", {})
            meta.setdefault("fromCache", False)
            meta.setdefault("cacheAge", 0)

        return result

    async def run(self, symbols: list[str], **kwargs: Any) -> dict[str, Any]:
        """Fetch all symbols and return a standard FinMCP-shaped envelope.

        Symbols are normalised (strip + upper) and deduplicated while
        preserving order.

        Return shape::

            {
                "results": {
                    "AAPL": {
                        "data": ...,
                        "meta": {"fromCache": false, "cacheAge": 0, "warnings": []}
                    },
                    ...
                },
                "summary": {
                    "totalRequested": <int>,
                    "totalReturned": <int>,
                    "errors": [{"symbol": "BAD", "error": "..."}, ...]
                }
            }
        """
        # Normalise and deduplicate (preserve order).
        seen: set[str] = set()
        unique: list[str] = []
        for sym in symbols:
            normalised = sym.strip().upper()
            if normalised not in seen:
                seen.add(normalised)
                unique.append(normalised)

        results: dict[str, Any] = {}
        errors: list[dict[str, str]] = []

        batches = [unique[i : i + self._batch_size] for i in range(0, len(unique), self._batch_size)]

        for batch_idx, batch in enumerate(batches):
            batch_results = await asyncio.gather(*[self._fetch_one(sym, kwargs) for sym in batch])

            for sym, result in zip(batch, batch_results, strict=True):
                if "error" in result:
                    errors.append({"symbol": sym, "error": result["error"]})
                else:
                    results[sym] = result

            if batch_idx < len(batches) - 1:
                await asyncio.sleep(self._batch_delay)

        return {
            "results": results,
            "summary": {
                "totalRequested": len(unique),
                "totalReturned": len(results),
                "errors": errors,
            },
        }
