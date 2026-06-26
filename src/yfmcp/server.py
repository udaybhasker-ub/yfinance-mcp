import asyncio
import base64
import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Annotated
from typing import Any

import yfinance as yf
from loguru import logger
from mcp.server.auth.settings import AuthSettings
from mcp.server.auth.settings import ClientRegistrationOptions
from mcp.server.fastmcp import FastMCP
from mcp.types import Icon
from mcp.types import ImageContent
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl
from pydantic import Field

from yfmcp.analyst_fetcher import processor as analyst_processor
from yfmcp.auth import SharedSecretOAuthProvider
from yfmcp.chart import generate_chart
from yfmcp.earnings_fetcher import processor as earnings_processor
from yfmcp.financials_fetcher import processor as financials_processor
from yfmcp.industry import SECTOR_INDUSTY_MAPPING
from yfmcp.industry import _gather_industry_tables
from yfmcp.industry import _sector_key
from yfmcp.jq_filter import TEMPLATE_FIELD_DESCRIPTION
from yfmcp.jq_filter import jq_or_json
from yfmcp.logging import _logged_tool
from yfmcp.news_fetcher import processor as news_processor
from yfmcp.options import _create_option_chain_fetch_error
from yfmcp.options import _create_option_dates_fetch_error
from yfmcp.options import _fetch_option_chain_for_date
from yfmcp.price_history_fetcher import processor as price_history_processor
from yfmcp.quote_fetcher import QuoteFetcher
from yfmcp.screener import build_screener_query
from yfmcp.screener import filter_us_currency_quotes
from yfmcp.screener import truncate_quotes
from yfmcp.types import ChartType
from yfmcp.types import Interval
from yfmcp.types import OptionChainType
from yfmcp.types import Period
from yfmcp.types import ScreenerQueryType
from yfmcp.types import SearchType
from yfmcp.types import Sector
from yfmcp.types import TopType
from yfmcp.utils import create_error_response
from yfmcp.yf_runner import _RETRYABLE_YFINANCE_EXCEPTIONS
from yfmcp.yf_runner import _create_retryable_error_response
from yfmcp.yf_runner import _get_ticker
from yfmcp.yf_runner import _is_retryable_yfinance_error
from yfmcp.yf_runner import _run_yf
from yfmcp.yf_runner import _select_retryable_exception

_EQUITY_FILTER_OVERFETCH_FACTOR = 4
_YAHOO_SCREENER_MAX_SIZE = 250

# ---------------------------------------------------------------------------
# Reusable Annotated type aliases for jq template parameters.
# Added to every JSON-returning tool so callers can reshape responses without
# a separate post-processing step.
# ---------------------------------------------------------------------------
_JqTemplate = Annotated[str | None, Field(description=TEMPLATE_FIELD_DESCRIPTION)]

_base_url = os.environ.get("MCP_PUBLIC_URL") or f"https://{os.environ.get('RAILWAY_PUBLIC_DOMAIN', 'localhost')}"

# MCP_AUTH_SECRET gates the OAuth login form in yfmcp.auth, letting this server be added
# as an authenticated connector (e.g. in Claude) instead of being open to anyone with the URL.
_auth_secret = os.environ.get("MCP_AUTH_SECRET")
_auth_provider = None
_auth_settings = None
if _auth_secret:
    _public_url = AnyHttpUrl(_base_url)
    _auth_provider = SharedSecretOAuthProvider(_auth_secret)
    _auth_settings = AuthSettings(
        issuer_url=_public_url,
        resource_server_url=_public_url,
        client_registration_options=ClientRegistrationOptions(enabled=True),
    )

# https://github.com/jlowin/fastmcp/issues/81#issuecomment-2714245145
mcp = FastMCP(
    "yfinance_mcp",
    log_level="ERROR",
    auth_server_provider=_auth_provider,
    auth=_auth_settings,
    icons=[Icon(src=f"{_base_url}/favicon.png", mimeType="image/png", sizes=["256x256"])],
)


if _auth_provider is not None:

    @mcp.custom_route("/login", methods=["GET", "POST"], name="login", include_in_schema=False)
    async def login(request: Any) -> Any:
        return await _auth_provider.handle_login(request)


@mcp.custom_route("/health", methods=["GET"], name="health", include_in_schema=False)
async def health_check(request: Any) -> Any:
    """Health check endpoint for Railway and other platforms."""
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok", "service": "yfinance-mcp"})


_ASSETS_DIR = Path(__file__).parent / "assets"
_ERROR_HTTP_STATUS = {
    "INVALID_PARAMS": 400,
    "INVALID_SYMBOL": 404,
    "NO_DATA": 404,
    "NETWORK_ERROR": 503,
    "API_ERROR": 502,
    "UNKNOWN_ERROR": 500,
}


@mcp.custom_route("/favicon.ico", methods=["GET"], name="favicon_ico", include_in_schema=False)
async def favicon_ico(request: Any) -> Any:
    from starlette.responses import FileResponse

    return FileResponse(_ASSETS_DIR / "favicon.ico", media_type="image/x-icon")


@mcp.custom_route("/favicon.png", methods=["GET"], name="favicon_png", include_in_schema=False)
async def favicon_png(request: Any) -> Any:
    from starlette.responses import FileResponse

    return FileResponse(_ASSETS_DIR / "favicon.png", media_type="image/png")


def _parse_csv_query_values(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        for part in value.split(","):
            item = part.strip()
            if item:
                items.append(item)
    return items


def _get_query_list(request: Any, name: str) -> list[str]:
    return _parse_csv_query_values(request.query_params.getlist(name))


def _get_query_bool(request: Any, name: str, default: bool = False) -> bool:
    raw = request.query_params.get(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Invalid boolean value for '{name}': {raw}")


def _parse_json_tool_result(result: str) -> tuple[int, str]:
    payload = json.loads(result)
    status_code = 200
    if isinstance(payload, dict) and "error_code" in payload:
        status_code = _ERROR_HTTP_STATUS.get(str(payload["error_code"]), 500)
    return status_code, result


async def _rest_response(result: str | ImageContent) -> Any:
    from starlette.responses import Response

    if isinstance(result, ImageContent):
        return Response(content=base64.b64decode(result.data), media_type=result.mimeType)

    status_code, body = _parse_json_tool_result(result)
    return Response(content=body, status_code=status_code, media_type="application/json")


def _invalid_query_param_response(name: str, value: str, expected: str) -> str:
    return create_error_response(
        f"Invalid query parameter '{name}' with value '{value}'. Expected {expected}.",
        error_code="INVALID_PARAMS",
        details={"parameter": name, "value": value, "expected": expected},
    )


async def _unauthorized_rest_response() -> Any:
    from starlette.responses import JSONResponse

    return JSONResponse(
        {"error": "invalid_token", "error_description": "Authentication required"},
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer error="invalid_token", error_description="Authentication required"'},
    )


async def _authorize_rest_request(request: Any) -> Any | None:
    if _auth_provider is None:
        return None

    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return await _unauthorized_rest_response()

    token = auth_header[7:].strip()
    access_token = await _auth_provider.load_access_token(token)
    if access_token is None:
        return await _unauthorized_rest_response()

    return None


def _protected_custom_route(path: str, methods: list[str], name: str):
    def decorator(func):
        @wraps(func)
        async def wrapped(request: Any) -> Any:
            auth_error = await _authorize_rest_request(request)
            if auth_error is not None:
                return auth_error
            return await func(request)

        return mcp.custom_route(path, methods=methods, name=name, include_in_schema=False)(wrapped)

    return decorator


@_protected_custom_route("/ticker/{symbol}", methods=["GET"], name="rest_ticker_info")
async def rest_get_ticker_info(request: Any) -> Any:
    return await _rest_response(await get_ticker_info(request.path_params["symbol"]))


@_protected_custom_route("/quote", methods=["GET"], name="rest_quote")
async def rest_get_quote(request: Any) -> Any:
    symbols = _get_query_list(request, "symbols")
    fields = _get_query_list(request, "fields") or None
    no_cache = _get_query_bool(request, "no_cache", False)

    if not symbols:
        return await _rest_response(
            create_error_response(
                "Query parameter 'symbols' is required.",
                error_code="INVALID_PARAMS",
                details={"parameter": "symbols"},
            )
        )

    return await _rest_response(await get_quote(symbols, fields=fields, no_cache=no_cache))


@_protected_custom_route("/quote/{symbol}", methods=["GET"], name="rest_quote_single")
async def rest_get_quote_single(request: Any) -> Any:
    fields = _get_query_list(request, "fields") or None
    no_cache = _get_query_bool(request, "no_cache", False)
    return await _rest_response(await get_quote([request.path_params["symbol"]], fields=fields, no_cache=no_cache))


@_protected_custom_route("/news", methods=["GET"], name="rest_news")
async def rest_get_news(request: Any) -> Any:
    symbols = _get_query_list(request, "symbols")
    no_cache = _get_query_bool(request, "no_cache", False)

    if not symbols:
        return await _rest_response(
            create_error_response(
                "Query parameter 'symbols' is required.",
                error_code="INVALID_PARAMS",
                details={"parameter": "symbols"},
            )
        )

    return await _rest_response(await get_ticker_news(symbols, no_cache=no_cache))


@_protected_custom_route("/news/{symbol}", methods=["GET"], name="rest_news_single")
async def rest_get_news_single(request: Any) -> Any:
    no_cache = _get_query_bool(request, "no_cache", False)
    return await _rest_response(await get_ticker_news([request.path_params["symbol"]], no_cache=no_cache))


@_protected_custom_route("/search", methods=["GET"], name="rest_search")
async def rest_search(request: Any) -> Any:
    query = request.query_params.get("q")
    search_type = request.query_params.get("type", "all")

    if not query:
        return await _rest_response(
            create_error_response(
                "Query parameter 'q' is required.",
                error_code="INVALID_PARAMS",
                details={"parameter": "q"},
            )
        )

    return await _rest_response(await search(query=query, search_type=search_type))


@_protected_custom_route("/screen", methods=["GET", "POST"], name="rest_screen")
async def rest_screen(request: Any) -> Any:
    if request.method == "GET":
        query = request.query_params.get("query")
        if not query:
            return await _rest_response(
                create_error_response(
                    "Query parameter 'query' is required for GET /screen.",
                    error_code="INVALID_PARAMS",
                    details={"parameter": "query", "method": "GET"},
                )
            )

        query_type = request.query_params.get("query_type", "predefined")
        offset_raw = request.query_params.get("offset")
        size_raw = request.query_params.get("size")
        count_raw = request.query_params.get("count")
        sort_field = request.query_params.get("sort_field")
        sort_asc_raw = request.query_params.get("sort_asc")
        user_id = request.query_params.get("user_id")
        user_id_type = request.query_params.get("user_id_type")

        try:
            offset = int(offset_raw) if offset_raw is not None else None
            size = int(size_raw) if size_raw is not None else None
            count = int(count_raw) if count_raw is not None else None
            sort_asc = None if sort_asc_raw is None else _get_query_bool(request, "sort_asc")
        except ValueError as exc:
            return await _rest_response(
                create_error_response(
                    str(exc),
                    error_code="INVALID_PARAMS",
                    details={"method": "GET"},
                )
            )

        return await _rest_response(
            await screen(
                query=query,
                query_type=query_type,
                offset=offset,
                size=size,
                count=count,
                sort_field=sort_field,
                sort_asc=sort_asc,
                user_id=user_id,
                user_id_type=user_id_type,
            )
        )

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return await _rest_response(
            create_error_response(
                "Request body must be valid JSON.",
                error_code="INVALID_PARAMS",
                details={"method": "POST"},
            )
        )

    return await _rest_response(
        await screen(
            query=payload.get("query"),
            query_type=payload.get("query_type", "predefined"),
            offset=payload.get("offset"),
            size=payload.get("size"),
            count=payload.get("count"),
            sort_field=payload.get("sort_field"),
            sort_asc=payload.get("sort_asc"),
            user_id=payload.get("user_id"),
            user_id_type=payload.get("user_id_type"),
        )
    )


@_protected_custom_route("/screen/gappers", methods=["GET"], name="rest_screen_gappers")
async def rest_screen_gappers(request: Any) -> Any:
    try:
        min_percent_change = float(request.query_params.get("min_percent_change", "3.0"))
        min_price = float(request.query_params.get("min_price", "5.0"))
        min_volume = int(request.query_params.get("min_volume", "500000"))
        min_market_cap = int(request.query_params.get("min_market_cap", "2000000000"))
        region = request.query_params.get("region", "us")
        size = int(request.query_params.get("size", "50"))
        offset = int(request.query_params.get("offset", "0"))
        sort_asc = _get_query_bool(request, "sort_asc", False)
    except ValueError as exc:
        return await _rest_response(create_error_response(str(exc), error_code="INVALID_PARAMS"))

    return await _rest_response(
        await screen_gappers(
            min_percent_change=min_percent_change,
            min_price=min_price,
            min_volume=min_volume,
            min_market_cap=min_market_cap,
            region=region,
            size=size,
            offset=offset,
            sort_asc=sort_asc,
        )
    )


@_protected_custom_route("/top/{sector}", methods=["GET"], name="rest_top")
async def rest_get_top(request: Any) -> Any:
    top_type = request.query_params.get("type")
    if not top_type:
        return await _rest_response(
            create_error_response(
                "Query parameter 'type' is required.",
                error_code="INVALID_PARAMS",
                details={"parameter": "type"},
            )
        )

    top_n_raw = request.query_params.get("n", "10")
    try:
        top_n = int(top_n_raw)
    except ValueError:
        return await _rest_response(_invalid_query_param_response("n", top_n_raw, "an integer"))

    return await _rest_response(await get_top(sector=request.path_params["sector"], top_type=top_type, top_n=top_n))


@_protected_custom_route("/price-history", methods=["GET"], name="rest_price_history")
async def rest_get_price_history(request: Any) -> Any:
    symbols = _get_query_list(request, "symbols")
    if not symbols:
        return await _rest_response(
            create_error_response(
                "Query parameter 'symbols' is required.",
                error_code="INVALID_PARAMS",
                details={"parameter": "symbols"},
            )
        )

    period = request.query_params.get("period", "1mo")
    interval = request.query_params.get("interval", "1d")
    chart_type = request.query_params.get("chart_type")
    prepost = _get_query_bool(request, "prepost", False)
    no_cache = _get_query_bool(request, "no_cache", False)

    return await _rest_response(
        await get_price_history(
            symbols=symbols,
            period=period,
            interval=interval,
            chart_type=chart_type,
            prepost=prepost,
            no_cache=no_cache,
        )
    )


@_protected_custom_route("/financials", methods=["GET"], name="rest_financials")
async def rest_get_financials(request: Any) -> Any:
    symbols = _get_query_list(request, "symbols")
    no_cache = _get_query_bool(request, "no_cache", False)
    frequency = request.query_params.get("frequency", "annual")

    if not symbols:
        return await _rest_response(
            create_error_response(
                "Query parameter 'symbols' is required.",
                error_code="INVALID_PARAMS",
                details={"parameter": "symbols"},
            )
        )

    return await _rest_response(await get_financials(symbols=symbols, frequency=frequency, no_cache=no_cache))


@_protected_custom_route("/options/{symbol}", methods=["GET"], name="rest_option_chain")
async def rest_get_option_chain(request: Any) -> Any:
    expiration_date = request.query_params.get("expiration_date")
    option_type = request.query_params.get("option_type", "all")
    return await _rest_response(
        await get_option_chain(
            symbol=request.path_params["symbol"],
            expiration_date=expiration_date,
            option_type=option_type,
        )
    )


@_protected_custom_route("/options/{symbol}/dates", methods=["GET"], name="rest_option_dates")
async def rest_get_option_dates(request: Any) -> Any:
    return await _rest_response(await get_option_dates(request.path_params["symbol"]))


@_protected_custom_route("/holders/{symbol}", methods=["GET"], name="rest_holders")
async def rest_get_holders(request: Any) -> Any:
    max_rows_raw = request.query_params.get("max_rows", "10")
    try:
        max_rows = int(max_rows_raw)
    except ValueError:
        return await _rest_response(_invalid_query_param_response("max_rows", max_rows_raw, "an integer"))

    return await _rest_response(await get_holders(symbol=request.path_params["symbol"], max_rows=max_rows))


@_protected_custom_route("/earnings", methods=["GET"], name="rest_earnings")
async def rest_get_earnings(request: Any) -> Any:
    symbols = _get_query_list(request, "symbols")
    no_cache = _get_query_bool(request, "no_cache", False)
    history_limit_raw = request.query_params.get("history_limit", "12")

    if not symbols:
        return await _rest_response(
            create_error_response(
                "Query parameter 'symbols' is required.",
                error_code="INVALID_PARAMS",
                details={"parameter": "symbols"},
            )
        )

    try:
        history_limit = int(history_limit_raw)
    except ValueError:
        return await _rest_response(_invalid_query_param_response("history_limit", history_limit_raw, "an integer"))

    return await _rest_response(
        await get_earnings(symbols=symbols, history_limit=history_limit, no_cache=no_cache)
    )


@_protected_custom_route("/analyst", methods=["GET"], name="rest_analyst")
async def rest_get_analyst(request: Any) -> Any:
    symbols = _get_query_list(request, "symbols")
    no_cache = _get_query_bool(request, "no_cache", False)
    upgrades_limit_raw = request.query_params.get("upgrades_limit", "20")

    if not symbols:
        return await _rest_response(
            create_error_response(
                "Query parameter 'symbols' is required.",
                error_code="INVALID_PARAMS",
                details={"parameter": "symbols"},
            )
        )

    try:
        upgrades_limit = int(upgrades_limit_raw)
    except ValueError:
        return await _rest_response(_invalid_query_param_response("upgrades_limit", upgrades_limit_raw, "an integer"))

    return await _rest_response(
        await get_analyst(symbols=symbols, upgrades_limit=upgrades_limit, no_cache=no_cache)
    )


@_protected_custom_route("/combined-quote", methods=["GET"], name="rest_combined_quote")
async def rest_get_combined_quote(request: Any) -> Any:
    symbols = _get_query_list(request, "symbols")
    quote_fields = _get_query_list(request, "quote_fields") or None
    no_cache = _get_query_bool(request, "no_cache", False)
    history_limit_raw = request.query_params.get("history_limit", "8")
    upgrades_limit_raw = request.query_params.get("upgrades_limit", "10")

    if not symbols:
        return await _rest_response(
            create_error_response(
                "Query parameter 'symbols' is required.",
                error_code="INVALID_PARAMS",
                details={"parameter": "symbols"},
            )
        )

    try:
        history_limit = int(history_limit_raw)
        upgrades_limit = int(upgrades_limit_raw)
    except ValueError:
        return await _rest_response(
            create_error_response(
                "Query parameters 'history_limit' and 'upgrades_limit' must be integers.",
                error_code="INVALID_PARAMS",
                details={
                    "history_limit": history_limit_raw,
                    "upgrades_limit": upgrades_limit_raw,
                },
            )
        )

    return await _rest_response(
        await get_combined_quote(
            symbols=symbols,
            quote_fields=quote_fields,
            history_limit=history_limit,
            upgrades_limit=upgrades_limit,
            no_cache=no_cache,
        )
    )


@mcp.tool(
    name="yfinance_get_ticker_info",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_ticker_info(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    template: _JqTemplate = None,
) -> str:
    """Retrieve comprehensive stock data including company information, financials, trading metrics and governance.

    Returns JSON object with fields including:
    - Company: symbol, longName, sector, industry, longBusinessSummary, website, city, country
    - Price: currentPrice, previousClose, open, dayHigh, dayLow, fiftyTwoWeekHigh, fiftyTwoWeekLow
    - Valuation: marketCap, enterpriseValue, trailingPE, forwardPE, priceToBook, pegRatio
    - Trading: volume, averageVolume, averageVolume10days, bid, ask, bidSize, askSize
    - Dividends: dividendRate, dividendYield, exDividendDate, payoutRatio
    - Financials: totalRevenue, revenueGrowth, earningsGrowth, profitMargins, operatingMargins
    - Performance: beta, fiftyDayAverage, twoHundredDayAverage, trailingEps, forwardEps

    Note: Available fields vary by security type. Timestamps are converted to readable dates.
    """
    try:
        ticker = await _run_yf(yf.Ticker, symbol)
        info = await _run_yf(lambda: ticker.info)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching ticker info for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch ticker info for '{symbol}'. Verify the symbol is correct and try again.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    if not info:
        return create_error_response(
            f"No information available for symbol '{symbol}'. "
            "The symbol may be invalid or delisted. Try searching for the company "
            "name using the 'yfinance_search' tool to find the correct symbol.",
            error_code="INVALID_SYMBOL",
            details={"symbol": symbol},
        )

    # Convert timestamps to human-readable format when they look numeric.
    for key, value in list(info.items()):
        if not isinstance(key, str):
            continue

        if not isinstance(value, int | float):
            continue

        if key.lower().endswith(("date", "start", "end", "timestamp", "time", "quarter")):
            try:
                info[key] = datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as exc:
                logger.error("Unable to convert {}: {} to datetime: {}", key, value, exc)

    return jq_or_json(info, template)


@mcp.tool(
    name="yfinance_get_quote",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_quote(
    symbols: Annotated[
        list[str],
        Field(
            description=(
                "One or more stock ticker symbols, e.g. ['AAPL', 'NVDA', 'MSFT']. "
                "Up to 100 tickers per call. Processed in batches of 10 with a 200 ms "
                "delay between batches to avoid rate-limiting."
            ),
            min_length=1,
            max_length=100,
        ),
    ],
    fields: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional list of yfinance ticker.info field names to return. "
                "When omitted, a curated ~35-field set is used (see tool description). "
                "Example: ['currentPrice', 'trailingPE', 'targetMeanPrice', 'sector']"
            ),
        ),
    ] = None,
    no_cache: Annotated[
        bool,
        Field(description="Bypass the 1-minute cache and fetch fresh data. The result is written back to cache."),
    ] = False,
    template: _JqTemplate = None,
) -> str:
    """Batch quote fetch for one or more tickers with an analysis-ready field set.

    Returns a curated ~35-field subset of yfinance ticker.info per ticker (vs the full
    ~127-field blob), keeping payloads small so parallel multi-ticker calls are reliable.
    Pass the optional `fields` list to override the defaults.

    Default fields:
    - Price: currentPrice, regularMarketChange, regularMarketChangePercent,
             regularMarketPreviousClose, regularMarketOpen,
             regularMarketDayRange {low,high}, fiftyTwoWeekRange {low,high}
    - Volume: regularMarketVolume, averageVolume, averageVolume10days
    - Valuation: marketCap, trailingPE, forwardPE, trailingEps, forwardEps, pegRatio, priceToBook
    - Dividends: dividendRate, dividendYield, exDividendDate
    - Technicals: beta, fiftyDayAverage, twoHundredDayAverage
    - Shares: sharesOutstanding
    - Analyst: targetMeanPrice, recommendationKey, numberOfAnalystOpinions
    - Earnings: earningsTimestamp
    - Growth: revenueGrowth, earningsGrowth, profitMargins
    - Identity: longName, sector, industry

    Response shape (mirrors FinMCP get_quote):
      {
        "results": {
          "AAPL": { "data": { ... }, "meta": { "dataAge": 0, "completenessScore": 0.97, "warnings": [] } }
        },
        "summary": { "totalRequested": 1, "totalReturned": 1, "errors": [] }
      }
    """
    return jq_or_json(await QuoteFetcher.fetch_batch(symbols, fields, no_cache=no_cache), template)


@mcp.tool(
    name="yfinance_get_ticker_news",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_ticker_news(
    symbols: Annotated[
        list[str],
        Field(
            description=(
                "One or more stock ticker symbols, e.g. ['AAPL', 'GOOGL']. "
                "Up to 20 tickers per call. Results cached for 2 minutes."
            ),
            min_length=1,
            max_length=20,
        ),
    ],
    no_cache: Annotated[
        bool,
        Field(description="Bypass the 2-minute cache and fetch fresh headlines. The result is written back to cache."),
    ] = False,
    template: _JqTemplate = None,
) -> str:
    """Fetch recent news articles and press releases for one or more stocks.

    Returns a batched envelope keyed by ticker:

        {
          "results": {
            "AAPL": {
              "data": [<article>, ...],
              "meta": {"articleCount": 8, "fromCache": false, "cacheAge": 0, "warnings": []}
            }
          },
          "summary": {"totalRequested": 1, "totalReturned": 1, "errors": []}
        }

    Each article object contains:
    - id: Unique article identifier
    - content.title: Article headline
    - content.summary: Brief article summary
    - content.pubDate: Publication date (ISO 8601)
    - content.provider.displayName: News source name
    - content.canonicalUrl.url: Article URL
    - content.contentType: "STORY", "VIDEO", etc.

    Use this to track company announcements, market sentiment, and breaking news.
    """
    return jq_or_json(await news_processor.run(symbols, no_cache=no_cache), template)


@mcp.tool(
    name="yfinance_search",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def search(
    query: Annotated[str, Field(description="Search query - company name, ticker symbol, or keywords")],
    search_type: Annotated[
        SearchType,
        Field(
            description="Filter results: 'all' (quotes + news), 'quotes' (stocks/ETFs only), or 'news' (articles only)"
        ),
    ],
    template: _JqTemplate = None,
) -> str:
    """Search Yahoo Finance for stocks, ETFs, and news articles.

    Returns JSON with search results based on search_type:

    - 'quotes': Array of securities with:
        - symbol: Ticker symbol
        - shortname/longname: Company name
        - quoteType: Security type (EQUITY, ETF, MUTUALFUND, etc.)
        - exchange: Exchange code
        - sector: Business sector
        - industry: Industry classification
        - score: Search relevance score

    - 'news': Array of articles with:
        - uuid: Article identifier
        - title: Headline
        - publisher: News source
        - link: Article URL
        - providerPublishTime: Unix timestamp
        - relatedTickers: Array of related symbols
        - thumbnail: Image URLs

    - 'all': Object with both 'quotes' and 'news' arrays

    Use this to find ticker symbols, discover related securities, or search financial news.
    """
    try:
        s = await _run_yf(yf.Search, query)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"searching for '{query}'", exc, {"query": query})
    except Exception as exc:
        return create_error_response(
            f"Search failed for '{query}'. Try simplifying your query or using different keywords.",
            error_code="API_ERROR",
            details={"query": query, "exception": str(exc)},
        )

    match search_type.lower():
        case "all":
            return jq_or_json(s.all, template)
        case "quotes":
            return jq_or_json(s.quotes, template)
        case "news":
            return jq_or_json(s.news, template)
        case _:
            return create_error_response(
                f"Invalid search_type '{search_type}'. Valid options: 'all', 'quotes', 'news'.",
                error_code="INVALID_PARAMS",
                details={"search_type": search_type, "valid_options": ["all", "quotes", "news"]},
            )


@mcp.tool(
    name="yfinance_screen",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def screen(
    query: Annotated[
        str | dict[str, Any],
        Field(
            description=(
                "Screener query. For query_type='predefined': string key like 'day_gainers'. "
                "For query_type='equity' or 'fund': query tree object with {operator, operands} nodes."
            )
        ),
    ],
    query_type: Annotated[
        ScreenerQueryType,
        Field(description="Query mode: 'predefined', 'equity', or 'fund'."),
    ] = "predefined",
    offset: Annotated[int | None, Field(description="Result offset.", ge=0)] = None,
    size: Annotated[
        int | None,
        Field(description="Rows to return for custom queries. Yahoo maximum is 250.", ge=1, le=250),
    ] = None,
    count: Annotated[
        int | None,
        Field(description="Rows to return for predefined queries. Yahoo maximum is 250.", ge=1, le=250),
    ] = None,
    sort_field: Annotated[str | None, Field(description="Sort field, for example 'percentchange'.")] = None,
    sort_asc: Annotated[bool | None, Field(description="Sort ascending if true, descending if false.")] = None,
    user_id: Annotated[str | None, Field(description="Optional Yahoo user id.")] = None,
    user_id_type: Annotated[str | None, Field(description="Optional Yahoo user id type, commonly 'guid'.")] = None,
    template: _JqTemplate = None,
) -> str:
    """Run a Yahoo Finance screener query.

    Supports predefined Yahoo screener keys and custom equity or fund query trees.
    """
    try:
        if query_type == "predefined" and size is not None:
            return create_error_response(
                "For query_type='predefined', use count instead of size.",
                error_code="INVALID_PARAMS",
                details={"query_type": query_type, "invalid_parameter": "size", "expected_parameter": "count"},
            )
        if query_type in {"equity", "fund"} and count is not None:
            return create_error_response(
                "For query_type='equity' or 'fund', use size instead of count.",
                error_code="INVALID_PARAMS",
                details={"query_type": query_type, "invalid_parameter": "count", "expected_parameter": "size"},
            )

        if query_type == "predefined":
            if not isinstance(query, str):
                return create_error_response(
                    "For query_type='predefined', query must be a string screener key.",
                    error_code="INVALID_PARAMS",
                    details={"query_type": query_type, "expected_query_type": "string"},
                )

            predefined = getattr(yf, "PREDEFINED_SCREENER_QUERIES", {})
            if query not in predefined:
                return create_error_response(
                    f"Unknown predefined screener '{query}'.",
                    error_code="INVALID_PARAMS",
                    details={
                        "query": query,
                        "query_type": query_type,
                        "valid_predefined_queries": sorted(predefined.keys()),
                    },
                )

            resolved_query: str | Any = query
        else:
            if not isinstance(query, dict):
                return create_error_response(
                    "For query_type='equity' or 'fund', query must be an object with 'operator' and 'operands'.",
                    error_code="INVALID_PARAMS",
                    details={"query_type": query_type, "expected_query_type": "object"},
                )

            resolved_query = build_screener_query(query_type=query_type, query=query)

        fetch_size = size
        if query_type == "equity" and size is not None:
            fetch_size = min(size * _EQUITY_FILTER_OVERFETCH_FACTOR, _YAHOO_SCREENER_MAX_SIZE)

        result = await _run_yf(
            yf.screen,
            resolved_query,
            offset=offset,
            size=fetch_size,
            count=count,
            sortField=sort_field,
            sortAsc=sort_asc,
            userId=user_id,
            userIdType=user_id_type,
        )
    except (TypeError, ValueError) as exc:
        return create_error_response(
            "Invalid screener query. Check operators, operands, and field values for the selected query_type.",
            error_code="INVALID_PARAMS",
            details={"query_type": query_type, "exception": str(exc)},
        )
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response("running screener query", exc, {"query_type": query_type})
    except Exception as exc:
        return create_error_response(
            "Failed to run screener query.",
            error_code="API_ERROR",
            details={"query_type": query_type, "exception": str(exc)},
        )

    if query_type == "equity":
        result = filter_us_currency_quotes(result)
        if size is not None:
            result = truncate_quotes(result, size)

    return jq_or_json(result, template)


@mcp.tool(
    name="yfinance_screen_gappers",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def screen_gappers(
    min_percent_change: Annotated[
        float,
        Field(description="Minimum percent change from prior close, for example 3.0 for +3%.", ge=0),
    ] = 3.0,
    min_price: Annotated[
        float,
        Field(description="Minimum current intraday price.", ge=0),
    ] = 5.0,
    min_volume: Annotated[
        int,
        Field(description="Minimum intraday trading volume.", ge=0),
    ] = 500000,
    min_market_cap: Annotated[
        int,
        Field(description="Minimum intraday market cap in USD.", ge=0),
    ] = 2000000000,
    region: Annotated[
        str,
        Field(description="Yahoo screener region code, for example 'us'."),
    ] = "us",
    size: Annotated[
        int,
        Field(description="Rows to return. Yahoo maximum is 250.", ge=1, le=250),
    ] = 50,
    offset: Annotated[
        int,
        Field(description="Result offset for pagination.", ge=0),
    ] = 0,
    sort_asc: Annotated[
        bool,
        Field(description="Sort by percentchange ascending if true, descending if false."),
    ] = False,
    template: _JqTemplate = None,
) -> str:
    """Run a custom equity screener tuned for opening-session stock gappers."""
    query = {
        "operator": "and",
        "operands": [
            {"operator": "gte", "operands": ["percentchange", min_percent_change]},
            {"operator": "eq", "operands": ["region", region]},
            {"operator": "gte", "operands": ["intradaymarketcap", min_market_cap]},
            {"operator": "gte", "operands": ["intradayprice", min_price]},
            {"operator": "gte", "operands": ["dayvolume", min_volume]},
        ],
    }

    try:
        resolved_query = build_screener_query(query_type="equity", query=query)
        fetch_size = min(size * _EQUITY_FILTER_OVERFETCH_FACTOR, _YAHOO_SCREENER_MAX_SIZE)
        result = await _run_yf(
            yf.screen,
            resolved_query,
            offset=offset,
            size=fetch_size,
            sortField="percentchange",
            sortAsc=sort_asc,
        )
    except (TypeError, ValueError) as exc:
        return create_error_response(
            "Invalid gappers screener parameters.",
            error_code="INVALID_PARAMS",
            details={"exception": str(exc)},
        )
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response("running gappers screener", exc, {})
    except Exception as exc:
        return create_error_response(
            "Failed to run gappers screener.",
            error_code="API_ERROR",
            details={"exception": str(exc)},
        )

    result = filter_us_currency_quotes(result)
    result = truncate_quotes(result, size)

    return jq_or_json(result, template)


async def get_top_etfs(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top ETFs to retrieve", ge=1)],
    template: str | None = None,
) -> str:
    """Get the most popular ETFs for a specific sector.

    Returns JSON array where each ETF has:
    - symbol: ETF ticker symbol
    - name: Full ETF name
    """
    try:
        s = await _run_yf(yf.Sector, _sector_key(sector))
        etfs = await _run_yf(lambda: s.top_etfs)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching top ETFs for '{sector}'", exc, {"sector": sector})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch top ETFs for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if not etfs:
        return create_error_response(
            f"No ETF data available for sector '{sector}'.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    result = [{"symbol": symbol, "name": name} for symbol, name in list(etfs.items())[:top_n]]
    return jq_or_json(result, template)


async def get_top_mutual_funds(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top mutual funds to retrieve", ge=1)],
    template: str | None = None,
) -> str:
    """Get the most popular mutual funds for a specific sector.

    Returns JSON array where each mutual fund has:
    - symbol: Fund ticker symbol
    - name: Full fund name
    """
    try:
        s = await _run_yf(yf.Sector, _sector_key(sector))
        funds = await _run_yf(lambda: s.top_mutual_funds)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(
            f"fetching top mutual funds for '{sector}'",
            exc,
            {"sector": sector},
        )
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch top mutual funds for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if not funds:
        return create_error_response(
            f"No mutual fund data available for sector '{sector}'.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    result = [{"symbol": symbol, "name": name} for symbol, name in list(funds.items())[:top_n]]
    return jq_or_json(result, template)


async def get_top_companies(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top companies to retrieve", ge=1)],
    template: str | None = None,
) -> str:
    """Get top companies in a sector by market capitalization.

    Returns JSON array with company data from Yahoo Finance sector data.
    Typically includes company identifiers, market metrics, and analyst information.
    """
    try:
        s = await _run_yf(yf.Sector, _sector_key(sector))
        df = await _run_yf(lambda: s.top_companies)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching top companies for '{sector}'", exc, {"sector": sector})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch top companies for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if df is None or df.empty:
        return create_error_response(
            f"No company data available for '{sector}'. This sector may not have enough listed companies.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return jq_or_json(df.head(top_n).reset_index().to_dict(orient="records"), template)


async def get_top_growth_companies(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top growth companies per industry", ge=1)],
    template: str | None = None,
) -> str:
    """Get fastest-growing companies organized by industry within a sector.

    Returns JSON array grouped by industry. Each industry entry contains company data
    with growth-related metrics from Yahoo Finance.

    Results are organized by industry to show growth leaders across the sector.
    """
    try:
        industries = SECTOR_INDUSTY_MAPPING[sector]
    except KeyError:
        return create_error_response(
            f"Unknown sector '{sector}'. Valid sectors: {', '.join(SECTOR_INDUSTY_MAPPING.keys())}",
            error_code="INVALID_PARAMS",
            details={"sector": sector, "valid_sectors": list(SECTOR_INDUSTY_MAPPING.keys())},
        )

    industries = list(industries)
    expected_sector_key = _sector_key(sector)
    tables = await _gather_industry_tables(industries, "top_growth_companies", expected_sector_key)

    results = []
    for industry_name, df in zip(industries, tables, strict=False):
        if df is None or df.empty:
            continue

        results.append(
            {
                "industry": industry_name,
                "top_growth_companies": df.head(top_n).reset_index().to_dict(orient="records"),
            }
        )

    if not results:
        return create_error_response(
            f"No growth company data available for '{sector}'. Try a different sector or check back later.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return jq_or_json(results, template)


async def get_top_performing_companies(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top performing companies per industry", ge=1)],
    template: str | None = None,
) -> str:
    """Get best-performing companies by stock price performance, organized by industry.

    Returns JSON array grouped by industry. Each industry entry contains company data
    with performance-related metrics from Yahoo Finance.

    Results are organized by industry to show top performers across the sector.
    """
    try:
        industries = SECTOR_INDUSTY_MAPPING[sector]
    except KeyError:
        return create_error_response(
            f"Unknown sector '{sector}'. Valid sectors: {', '.join(SECTOR_INDUSTY_MAPPING.keys())}",
            error_code="INVALID_PARAMS",
            details={"sector": sector, "valid_sectors": list(SECTOR_INDUSTY_MAPPING.keys())},
        )

    industries = list(industries)
    expected_sector_key = _sector_key(sector)
    tables = await _gather_industry_tables(industries, "top_performing_companies", expected_sector_key)

    results = []
    for industry_name, df in zip(industries, tables, strict=False):
        if df is None or df.empty:
            continue

        results.append(
            {
                "industry": industry_name,
                "top_performing_companies": df.head(top_n).reset_index().to_dict(orient="records"),
            }
        )

    if not results:
        return create_error_response(
            f"No performance data available for '{sector}'. Try a different sector or check back later.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return jq_or_json(results, template)


@mcp.tool(
    name="yfinance_get_top",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_top(
    sector: Annotated[
        Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare', 'Financial Services')")
    ],
    top_type: Annotated[
        TopType,
        Field(
            description=(
                "Type of entities to retrieve: "
                "'top_etfs' (sector ETFs), "
                "'top_mutual_funds' (sector mutual funds), "
                "'top_companies' (largest by market cap), "
                "'top_growth_companies' (fastest revenue/earnings growth), "
                "'top_performing_companies' (best stock price performance)"
            )
        ),
    ],
    top_n: Annotated[
        int,
        Field(
            description="Number of top entities to retrieve per category/industry",
            ge=1,
            le=100,
        ),
    ] = 10,
    template: _JqTemplate = None,
) -> str:
    """Get top-ranked financial entities within a sector.

    This unified tool provides access to various rankings:
    - ETFs and mutual funds focused on the sector
    - Largest companies by market capitalization
    - Fastest-growing companies by revenue/earnings
    - Best-performing stocks by price appreciation

    Returns JSON data with relevant metrics for each entity type.
    """
    match top_type:
        case "top_etfs":
            return await get_top_etfs(sector, top_n, template=template)
        case "top_mutual_funds":
            return await get_top_mutual_funds(sector, top_n, template=template)
        case "top_companies":
            return await get_top_companies(sector, top_n, template=template)
        case "top_growth_companies":
            return await get_top_growth_companies(sector, top_n, template=template)
        case "top_performing_companies":
            return await get_top_performing_companies(sector, top_n, template=template)
        case _:
            return create_error_response(
                f"Invalid top_type '{top_type}'. "
                "Valid options: 'top_etfs', 'top_mutual_funds', 'top_companies', "
                "'top_growth_companies', 'top_performing_companies'.",
                error_code="INVALID_PARAMS",
                details={
                    "top_type": top_type,
                    "valid_options": [
                        "top_etfs",
                        "top_mutual_funds",
                        "top_companies",
                        "top_growth_companies",
                        "top_performing_companies",
                    ],
                },
            )


@mcp.tool(
    name="yfinance_get_price_history",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_price_history(
    symbols: Annotated[
        list[str],
        Field(
            description=(
                "One or more stock ticker symbols, e.g. ['AAPL', 'MSFT']. "
                "Up to 20 tickers per call. Results cached for 15 minutes. "
                "chart_type is only supported when exactly one symbol is provided."
            ),
            min_length=1,
            max_length=20,
        ),
    ],
    period: Annotated[
        Period,
        Field(
            description=(
                "Time range: '1d'/'5d' (days), '1mo'/'3mo'/'6mo' (months), "
                "'1y'/'2y'/'5y'/'10y' (years), 'ytd' (year-to-date), 'max' (all available data)"
            )
        ),
    ] = "1mo",
    interval: Annotated[
        Interval,
        Field(
            description=(
                "Data granularity: '1m'/'5m'/'15m'/'30m' (minutes), '1h' (hour), "
                "'1d'/'5d' (days), '1wk' (week), '1mo'/'3mo' (months). "
                "Short intervals require short periods (e.g., '1m' interval only works with '1d'/'5d' period)"
            )
        ),
    ] = "1d",
    chart_type: Annotated[
        ChartType | None,
        Field(
            description=(
                "Optional visualization (single-symbol only): "
                "'price_volume' (candlestick chart with volume bars), "
                "'vwap' (Volume Weighted Average Price overlay), "
                "'volume_profile' (volume distribution by price level). "
                "Ignored when multiple symbols are provided."
            )
        ),
    ] = None,
    prepost: Annotated[
        bool,
        Field(description="Include pre-market and post-market data when available"),
    ] = False,
    no_cache: Annotated[
        bool,
        Field(description="Bypass the 15-minute cache and fetch fresh data. The result is written back to cache."),
    ] = False,
    template: _JqTemplate = None,
) -> str | ImageContent:
    """Fetch historical price data for one or more tickers, with optional chart for single-ticker calls.

    Single-ticker mode (one symbol):
    - When chart_type is None, returns a JSON envelope with a Markdown table in ``data``.
    - When chart_type is set, returns the chart image directly (ImageContent).

    Multi-ticker mode (two or more symbols):
    - Always returns a JSON envelope keyed by ticker; chart_type is ignored.
    - Each ticker's ``data`` field contains its Markdown OHLCV table.

    Markdown table columns: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits.

    Chart types (single-symbol only):
    - 'price_volume': Candlestick chart with volume bars
    - 'vwap': Price with Volume Weighted Average Price overlay
    - 'volume_profile': Volume distribution by price level

    Note: Minute intervals (1m, 5m, etc.) only work with short periods (1d, 5d).
    """
    # Single-symbol + chart_type: bypass the batch path and return an image.
    if len(symbols) == 1 and chart_type is not None:
        symbol = symbols[0].strip().upper()
        try:
            ticker = await _get_ticker(symbol)
            df = await _run_yf(
                ticker.history,
                period=period,
                interval=interval,
                prepost=prepost,
                rounding=True,
            )
        except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
            return _create_retryable_error_response(
                f"fetching price history for '{symbol}'",
                exc,
                {"symbol": symbol, "period": period, "interval": interval},
            )
        except Exception as exc:
            return create_error_response(
                f"Failed to fetch price history for '{symbol}'. "
                "Verify the symbol is correct and the period/interval combination is valid.",
                error_code="API_ERROR",
                details={"symbol": symbol, "period": period, "interval": interval, "exception": str(exc)},
            )
        if df.empty:
            return create_error_response(
                f"No price data for '{symbol}' with period='{period}' interval='{interval}'.",
                error_code="NO_DATA",
                details={"symbol": symbol, "period": period, "interval": interval},
            )
        return await asyncio.to_thread(generate_chart, symbol=symbol, df=df, chart_type=chart_type)

    # Batch path — tabular only.
    data = await price_history_processor.run(
        symbols, no_cache=no_cache, period=period, interval=interval, prepost=prepost
    )
    return jq_or_json(data, template)


@mcp.tool(
    name="yfinance_get_financials",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_financials(
    symbols: Annotated[
        list[str],
        Field(
            description=(
                "One or more stock ticker symbols, e.g. ['AAPL', 'MSFT']. "
                "Up to 10 tickers per call. Results cached for 6 hours."
            ),
            min_length=1,
            max_length=10,
        ),
    ],
    frequency: Annotated[
        str,
        Field(
            description=(
                "Reporting frequency: 'annual' for yearly, 'quarterly' for quarterly, "
                "or 'ttm' for trailing twelve months"
            )
        ),
    ] = "annual",
    no_cache: Annotated[
        bool,
        Field(description="Bypass the 6-hour cache and fetch fresh statements. The result is written back to cache."),
    ] = False,
    template: _JqTemplate = None,
) -> str:
    """Fetch financial statements for one or more tickers (income statement, balance sheet, cash flow).

    Returns a batched envelope keyed by ticker:

        {
          "results": {
            "AAPL": {
              "data": {
                "income_statement": {"Total Revenue": {"2024-09-28": 391035000000, ...}, ...},
                "balance_sheet": {"Total Assets": {...}, ...},
                "cash_flow": {"Free Cash Flow": {...}, ...}
              },
              "meta": {"frequency": "annual", "fromCache": false, "cacheAge": 0, "warnings": []}
            }
          },
          "summary": {"totalRequested": 1, "totalReturned": 1, "errors": []}
        }

    Use the data to analyze trends, calculate ratios, or compare companies across periods.
    Financials update once per quarter; responses are cached for 6 hours.
    """
    return jq_or_json(await financials_processor.run(symbols, no_cache=no_cache, frequency=frequency), template)


@mcp.tool(
    name="yfinance_get_option_chain",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_option_chain(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    expiration_date: Annotated[
        str | None,
        Field(
            description=(
                "Option expiration date in YYYY-MM-DD format. "
                "Use the 'yfinance_get_option_dates' tool to find available dates, "
                "or omit to fetch all available expiration dates."
            )
        ),
    ] = None,
    option_type: Annotated[
        OptionChainType,
        Field(description=("Which options to return: 'calls', 'puts', or 'all' (both calls and puts).")),
    ] = "all",
    template: _JqTemplate = None,
) -> str:
    """Fetch option chain data (calls and puts) for a stock with available strike prices.

    Returns JSON with calls and/or puts data for each expiration date.

    JSON fields include:
    - contractSymbol: Option contract identifier
    - strike: Strike price
    - lastPrice: Last traded price
    - bid/ask: Bid and ask prices
    - volume: Trading volume
    - openInterest: Open interest
    - impliedVolatility: Implied volatility (IV)
    - inTheMoney: Whether option is ITM
    - contractSize: Contract size (REGULAR)
    - currency: Currency (USD)

    Use this to analyze options pricing, IV surfaces, and strike levels.
    """
    try:
        ticker = await _run_yf(yf.Ticker, symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching options for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch options for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    try:
        available_dates = await _run_yf(lambda: ticker.options)
    except Exception as exc:
        return _create_option_dates_fetch_error(
            symbol,
            exc,
            f"Failed to fetch option dates for '{symbol}'. The symbol may not have options.",
        )

    if not available_dates:
        return create_error_response(
            f"No options available for symbol '{symbol}'. "
            "This symbol may not have listed options (e.g., ETFs, stocks without options).",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    if expiration_date is not None and expiration_date not in available_dates:
        return create_error_response(
            f"Invalid expiration date '{expiration_date}' for '{symbol}'. Valid dates: {', '.join(available_dates)}",
            error_code="INVALID_PARAMS",
            details={
                "symbol": symbol,
                "expiration_date": expiration_date,
                "valid_dates": available_dates,
            },
        )

    dates_to_fetch = [expiration_date] if expiration_date else list(available_dates)
    result: dict[str, Any] = {}
    fetch_errors: list[tuple[str, Exception]] = []

    for date in dates_to_fetch:
        try:
            date_result = await _fetch_option_chain_for_date(ticker, date, option_type)
        except Exception as exc:
            logger.warning("Failed to fetch option chain for {} {}: {}", symbol, date, exc)
            fetch_errors.append((date, exc))
            continue
        result.update(date_result)

    if result:
        return jq_or_json(result, template)

    if fetch_errors:
        return _create_option_chain_fetch_error(symbol, dates_to_fetch, fetch_errors)

    return create_error_response(
        f"No option data retrieved for '{symbol}'.",
        error_code="NO_DATA",
        details={"symbol": symbol, "dates_requested": list(dates_to_fetch)},
    )


@mcp.tool(
    name="yfinance_get_option_dates",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_option_dates(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    template: _JqTemplate = None,
) -> str:
    """Fetch available option expiration dates for a stock.

    Returns JSON array of expiration dates in YYYY-MM-DD format.

    Use these dates with the 'yfinance_get_option_chain' tool to fetch
    the options chain for a specific date.
    """
    try:
        ticker = await _run_yf(yf.Ticker, symbol)
        dates = await _run_yf(lambda: ticker.options)
    except Exception as exc:
        return _create_option_dates_fetch_error(
            symbol,
            exc,
            f"Failed to fetch option dates for '{symbol}'. Verify the symbol is correct.",
        )

    if not dates:
        return create_error_response(
            f"No options available for symbol '{symbol}'. "
            "This symbol may not have listed options (e.g., ETFs, stocks without options).",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    return jq_or_json(dates, template)


async def _fetch_holder_section(
    symbol: str,
    ticker: yf.Ticker,
    attr_name: str,
    result_key: str,
    result: dict[str, Any],
    section_metadata: dict[str, dict[str, int | bool]],
    fetch_errors: list[Exception],
    max_rows: int,
) -> None:
    """Fetch a single holder data section, adding successful data to result and failures to fetch_errors."""
    try:
        df = await _run_yf(lambda t=ticker: getattr(t, attr_name))
    except Exception as exc:
        logger.warning("Failed to fetch {} for {}: {}", attr_name, symbol, exc)
        fetch_errors.append(exc)
        return
    if df is not None and not df.empty:
        if attr_name == "major_holders":
            df = df.reset_index()

        total_rows = len(df)
        limited_df = df if max_rows == 0 else df.head(max_rows)
        limited_records = limited_df.to_dict(orient="records")
        result[result_key] = limited_records
        section_metadata[result_key] = {
            "total_rows": total_rows,
            "returned_rows": len(limited_records),
            "truncated": len(limited_records) < total_rows,
        }


@mcp.tool(
    name="yfinance_get_holders",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_holders(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    max_rows: Annotated[
        int,
        Field(description="Maximum rows returned per holder section. Use 0 to return all rows."),
    ] = 10,
    template: _JqTemplate = None,
) -> str:
    """Fetch major holders, institutional holders, mutual fund holders, and insider data.

    Returns JSON with:
    - major_holders: Aggregated breakdown including insider % held, institutional % held,
      institutional % float held, and institution count.
    - institutional_holders: List of institutional investors with shares held, date reported,
      value, and % change.
    - mutualfund_holders: List of mutual fund holders with same fields.
    - insider_transactions: Recent insider transactions including shares, value, transaction
      type, and date.
    - insider_purchases: Summary of insider buy/sell activity over the last 6 months.
    - insider_roster: List of known insiders by name and position.

    Use this to analyze ownership concentration, insider activity, and institutional interest.
    """
    if max_rows < 0:
        return create_error_response(
            "max_rows must be greater than or equal to 0.",
            error_code="INVALID_PARAMS",
            details={"max_rows": max_rows},
        )

    try:
        ticker = await _run_yf(yf.Ticker, symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching holders for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch holders for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    result: dict[str, Any] = {}
    section_metadata: dict[str, dict[str, int | bool]] = {}
    fetch_errors: list[Exception] = []
    await _fetch_holder_section(
        symbol, ticker, "major_holders", "major_holders", result, section_metadata, fetch_errors, max_rows
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "institutional_holders",
        "institutional_holders",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "mutualfund_holders",
        "mutualfund_holders",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "insider_transactions",
        "insider_transactions",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )
    await _fetch_holder_section(
        symbol, ticker, "insider_purchases", "insider_purchases", result, section_metadata, fetch_errors, max_rows
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "insider_roster_holders",
        "insider_roster",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )

    if not result:
        retryable_exceptions = [exc for exc in fetch_errors if _is_retryable_yfinance_error(exc)]
        if retryable_exceptions:
            return _create_retryable_error_response(
                f"fetching holders for '{symbol}'",
                _select_retryable_exception(retryable_exceptions),
                {"symbol": symbol},
            )

        return create_error_response(
            f"No holder data available for '{symbol}'. Verify the symbol is correct.",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    result["_metadata"] = {"max_rows": max_rows, "sections": section_metadata}
    return jq_or_json(result, template)


def main() -> None:
    import os

    from mcp.server.transport_security import TransportSecuritySettings

    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()

    if transport in ("streamable-http", "sse"):
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = int(os.environ.get("PORT", "8000"))
        # Server is fronted by Railway's edge proxy, which rewrites the Host header to the
        # public domain; the SDK's DNS-rebinding check rejects that by default, so disable it.
        mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

    if transport == "streamable-http":
        mcp.settings.streamable_http_path = "/mcp"
        mcp.run(transport="streamable-http")
    elif transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


@mcp.tool(
    name="yfinance_get_earnings",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_earnings(
    symbols: Annotated[
        list[str],
        Field(
            description=(
                "One or more stock ticker symbols, e.g. ['AAPL', 'NVDA']. "
                "Up to 20 tickers per call. Results cached for 1 hour."
            ),
            min_length=1,
            max_length=20,
        ),
    ],
    history_limit: Annotated[
        int,
        Field(
            description="Number of past/future earnings dates to return per ticker (max 100). Default 12 = ~3 years.",
            ge=1,
            le=100,
        ),
    ] = 12,
    no_cache: Annotated[
        bool,
        Field(description="Bypass the 1-hour cache and fetch fresh estimates. The result is written back to cache."),
    ] = False,
    template: _JqTemplate = None,
) -> str:
    """Fetch earnings beat/miss history, forward EPS/revenue estimates, and revision trends for one or more tickers.

    Returns a batched envelope keyed by ticker:

        {
          "results": {
            "AAPL": {
              "data": {
                "earnings_dates": {"2024-11-01 ...": {"EPS Estimate": 1.6, "Reported EPS": 1.64, "Surprise(%)": 2.5}},
                "earnings_estimate": {...},
                "revenue_estimate": {...},
                "eps_trend": {...},
                "eps_revisions": {...}
              },
              "meta": {"historyLimit": 12, "fromCache": false, "cacheAge": 0, "warnings": []}
            }
          },
          "summary": {"totalRequested": 1, "totalReturned": 1, "errors": []}
        }

    Useful for pre-earnings sweeps across a watchlist. Results cached for 1 hour.
    """
    return jq_or_json(await earnings_processor.run(symbols, no_cache=no_cache, history_limit=history_limit), template)


@mcp.tool(
    name="yfinance_get_analyst",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_analyst(
    symbols: Annotated[
        list[str],
        Field(
            description=(
                "One or more stock ticker symbols, e.g. ['AAPL', 'NVDA']. "
                "Up to 20 tickers per call. Results cached for 1 hour."
            ),
            min_length=1,
            max_length=20,
        ),
    ],
    upgrades_limit: Annotated[
        int,
        Field(description="Number of recent firm upgrades/downgrades to return per ticker. Default 20.", ge=1, le=200),
    ] = 20,
    no_cache: Annotated[
        bool,
        Field(description="Bypass the 1-hour cache and fetch fresh analyst data. The result is written back to cache."),
    ] = False,
    template: _JqTemplate = None,
) -> str:
    """Fetch analyst consensus, price targets, and upgrade/downgrade history for one or more tickers.

    Returns a batched envelope keyed by ticker:

        {
          "results": {
            "AAPL": {
              "data": {
                "price_targets": {"current": 195.0, "low": 160.0, "high": 260.0, "mean": 220.0, "median": 225.0},
                "recommendations": [{"period": "0m", "strongBuy": 20, "buy": 15, ...}, ...],
                "upgrades_downgrades": [{"GradeDate": "2024-11-01", "Firm": "...", "toGrade": "Buy", ...}, ...]
              },
              "meta": {"upgradesLimit": 20, "fromCache": false, "cacheAge": 0, "warnings": []}
            }
          },
          "summary": {"totalRequested": 1, "totalReturned": 1, "errors": []}
        }

    Useful for watchlist-wide consensus sweeps. Results cached for 1 hour.
    """
    return jq_or_json(await analyst_processor.run(symbols, no_cache=no_cache, upgrades_limit=upgrades_limit), template)


@mcp.tool(
    name="yfinance_get_combined_quote",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@_logged_tool
async def get_combined_quote(
    symbols: Annotated[
        list[str],
        Field(
            description=(
                "One or more stock ticker symbols, e.g. ['AAPL', 'NVDA']. "
                "Up to 20 tickers per call."
            ),
            min_length=1,
            max_length=20,
        ),
    ],
    quote_fields: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional list of yfinance ticker.info field names to include in the quote section. "
                "When omitted, the curated ~35-field default set is used."
            ),
        ),
    ] = None,
    history_limit: Annotated[
        int,
        Field(
            description="Number of past/future earnings dates to return per ticker (max 100). Default 8.",
            ge=1,
            le=100,
        ),
    ] = 8,
    upgrades_limit: Annotated[
        int,
        Field(description="Number of recent firm upgrades/downgrades to return per ticker. Default 10.", ge=1, le=200),
    ] = 10,
    no_cache: Annotated[
        bool,
        Field(description="Bypass all caches and fetch fresh data. Results are written back to cache."),
    ] = False,
    template: _JqTemplate = None,
) -> str:
    """Fetch quote + analyst + earnings for one or more tickers in a single call.

    Runs all three fetches concurrently and merges the results under a single envelope keyed by ticker.

    Response shape:
        {
          "results": {
            "AAPL": {
              "quote": { ...~35 price/valuation/identity fields... },
              "analyst": {
                "price_targets": {...},
                "recommendations": [...],
                "upgrades_downgrades": [...]
              },
              "earnings": {
                "earnings_dates": {...},
                "earnings_estimate": {...},
                "revenue_estimate": {...},
                "eps_trend": {...},
                "eps_revisions": {...}
              },
              "meta": {
                "quote": {...},
                "analyst": {...},
                "earnings": {...}
              }
            }
          },
          "summary": {"totalRequested": 1, "totalReturned": 1, "errors": []}
        }

    Use this instead of calling yfinance_get_quote + yfinance_get_analyst + yfinance_get_earnings separately.
    All three fetches run in parallel server-side; cached results are served from their respective caches.
    """
    quote_task = QuoteFetcher.fetch_batch(symbols, quote_fields, no_cache=no_cache)
    analyst_task = analyst_processor.run(symbols, no_cache=no_cache, upgrades_limit=upgrades_limit)
    earnings_task = earnings_processor.run(symbols, no_cache=no_cache, history_limit=history_limit)

    quote_res, analyst_res, earnings_res = await asyncio.gather(quote_task, analyst_task, earnings_task)

    merged: dict = {"results": {}, "summary": {"totalRequested": len(symbols), "totalReturned": 0, "errors": []}}

    all_symbols = (
        set(quote_res.get("results", {}).keys())
        | set(analyst_res.get("results", {}).keys())
        | set(earnings_res.get("results", {}).keys())
    )

    for sym in all_symbols:
        q = quote_res.get("results", {}).get(sym, {})
        a = analyst_res.get("results", {}).get(sym, {})
        e = earnings_res.get("results", {}).get(sym, {})

        merged["results"][sym] = {
            "quote": q.get("data", {}),
            "analyst": a.get("data", {}),
            "earnings": e.get("data", {}),
            "meta": {
                "quote": q.get("meta", {}),
                "analyst": a.get("meta", {}),
                "earnings": e.get("meta", {}),
            },
        }

    # Collect errors from all three sub-calls
    for sub in (quote_res, analyst_res, earnings_res):
        merged["summary"]["errors"].extend(sub.get("summary", {}).get("errors", []))

    merged["summary"]["totalReturned"] = len(merged["results"])

    return jq_or_json(merged, template)
