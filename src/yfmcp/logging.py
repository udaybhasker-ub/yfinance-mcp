"""Logging configuration and the ``_logged_tool`` decorator for MCP tool functions."""

from __future__ import annotations

import asyncio
import functools
import inspect
import sys
import time
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Log format — configure once at import time so every module that imports
# ``logger`` from loguru picks up the same sink and format.
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)


def _format_call_params(func, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    try:
        bound = inspect.signature(func).bind(*args, **kwargs)
    except TypeError:
        return str(kwargs or args)
    return ", ".join(f"{name}={value!r}" for name, value in bound.arguments.items())


def _logged_tool(func):
    """Decorator that logs every MCP tool invocation's params, outcome, and duration.

    Without this, only the warning/error paths inside individual tools ever logged
    anything, so successful calls (the vast majority) were invisible in Railway logs.
    """
    tool_name = func.__name__

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        params = _format_call_params(func, args, kwargs)
        start = time.monotonic()
        try:
            result = await func(*args, **kwargs)
        except asyncio.CancelledError:
            logger.warning(
                "tool={} params=({}) status=suspended duration={:.2f}s",
                tool_name,
                params,
                time.monotonic() - start,
            )
            raise
        except Exception as exc:
            logger.error(
                "tool={} params=({}) status=failed duration={:.2f}s error={}",
                tool_name,
                params,
                time.monotonic() - start,
                exc,
            )
            raise
        else:
            logger.info(
                "tool={} params=({}) status=success duration={:.2f}s",
                tool_name,
                params,
                time.monotonic() - start,
            )
            return result

    return wrapper
