"""Industry and sector fetch helpers for the yfinance_get_top_* tools.

yfinance's Industry/Sector objects fan out into many individual Yahoo Finance calls.
This module contains the concurrency controls, retry logic, and deadline handling
that keep those fan-outs from stalling the MCP client.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import yfinance as yf
from loguru import logger
from yfinance.const import SECTOR_INDUSTY_MAPPING  # noqa: F401 — re-exported for server.py

from yfmcp.yf_runner import _RETRYABLE_YFINANCE_EXCEPTIONS

# ---------------------------------------------------------------------------
# Key-normalisation helpers
# ---------------------------------------------------------------------------


def _sector_key(name: str) -> str:
    """Convert human-readable sector name to Yahoo Finance API key format."""
    return name.lower().replace(" ", "-")


def _industry_key(name: str) -> str:
    """Convert human-readable industry name to Yahoo Finance API key format.

    SECTOR_INDUSTY_MAPPING uses em dashes (—) and title case,
    but the API expects lowercase with regular hyphens.
    """
    return name.lower().replace("& ", "").replace("- ", "").replace(", ", " ").replace("—", "-").replace(" ", "-")


# ---------------------------------------------------------------------------
# Concurrency controls
# ---------------------------------------------------------------------------

_INDUSTRY_FETCH_CONCURRENCY = asyncio.Semaphore(4)
_INDUSTRY_FETCH_TIMEOUT_SECONDS = 5.0

# Dedicated pool so slow/throttled Yahoo calls for industry data can't starve
# the shared default executor used by every other tool via asyncio.to_thread.
_INDUSTRY_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="yf-industry")

_INDUSTRY_FAN_OUT_DEADLINE_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


async def _run_in_industry_executor(func, *args) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_INDUSTRY_EXECUTOR, func, *args)


async def _fetch_industry_table_once(industry_name: str, attr: str, expected_sector_key: str) -> Any:
    industry = await _run_in_industry_executor(yf.Industry, _industry_key(industry_name))
    table = await _run_in_industry_executor(lambda: getattr(industry, attr))
    sector_key = await _run_in_industry_executor(lambda: industry.sector_key)
    if sector_key != expected_sector_key:
        logger.warning(
            "Industry '{}' returned sector_key '{}', expected '{}'; skipping.",
            industry_name,
            sector_key,
            expected_sector_key,
        )
        return None
    return table


async def _fetch_industry_table(industry_name: str, attr: str, expected_sector_key: str) -> Any:
    """Fetch a per-industry table with one retry.

    Sectors with many industries (e.g. Industrials has 25) fan out into many Yahoo
    calls. Concurrency is capped via a semaphore and each attempt is bounded to
    ``_INDUSTRY_FETCH_TIMEOUT_SECONDS`` so a stalled call fails fast.

    Also verifies the fetched industry actually belongs to the requested sector.
    """
    async with _INDUSTRY_FETCH_CONCURRENCY:
        for attempt in range(2):
            try:
                return await asyncio.wait_for(
                    _fetch_industry_table_once(industry_name, attr, expected_sector_key),
                    timeout=_INDUSTRY_FETCH_TIMEOUT_SECONDS,
                )
            except (TimeoutError, *_RETRYABLE_YFINANCE_EXCEPTIONS) as exc:
                if attempt == 0:
                    await asyncio.sleep(0.5)
                    continue
                logger.warning("Failed to load industry {} after retry: {}", industry_name, exc)
                return None
            except Exception as exc:
                logger.warning("Failed to load industry {}: {}", industry_name, exc)
                return None
    return None


async def _gather_industry_tables(industries: list[str], attr: str, expected_sector_key: str) -> list[Any]:
    """Run per-industry fetches with a hard wall-clock deadline for the whole fan-out.

    Per-attempt timeouts in ``_fetch_industry_table`` only bound a single fetch once it
    starts running. If multiple get_top calls land concurrently they can queue behind each
    other, stretching aggregate wait time far beyond any single-call timeout. This caps the
    entire fan-out so the tool returns whatever finished in time rather than hanging until
    an external timeout (MCP client, platform edge proxy) kills it.
    """
    if not industries:
        return []

    tasks = [
        asyncio.ensure_future(_fetch_industry_table(industry_name, attr, expected_sector_key))
        for industry_name in industries
    ]
    done, pending = await asyncio.wait(tasks, timeout=_INDUSTRY_FAN_OUT_DEADLINE_SECONDS)
    for task in pending:
        task.cancel()
    if pending:
        logger.warning(
            "Industry fan-out deadline hit; {} of {} industries unfinished.",
            len(pending),
            len(tasks),
        )
    return [task.result() if task in done and not task.cancelled() else None for task in tasks]
