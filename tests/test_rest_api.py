"""Tests for REST endpoints layered on top of MCP tools."""

import base64
from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from mcp.types import ImageContent

from yfmcp.auth import SharedSecretOAuthProvider
from yfmcp.server import mcp


@pytest_asyncio.fixture
async def rest_client():
    app = mcp.streamable_http_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


def test_rest_routes_are_registered() -> None:
    app = mcp.streamable_http_app()
    paths = {route.path for route in app.routes}

    expected_paths = {
        "/ticker/{symbol}",
        "/quote",
        "/quote/{symbol}",
        "/news",
        "/news/{symbol}",
        "/search",
        "/screen",
        "/screen/gappers",
        "/top/{sector}",
        "/price-history",
        "/financials",
        "/options/{symbol}",
        "/options/{symbol}/dates",
        "/holders/{symbol}",
        "/earnings",
        "/analyst",
        "/combined-quote",
    }

    assert expected_paths.issubset(paths)


@pytest.mark.asyncio
@patch("yfmcp.server.get_quote", new_callable=AsyncMock)
async def test_quote_route_supports_csv_and_repeated_query_params(
    mock_get_quote: AsyncMock, rest_client: httpx.AsyncClient
) -> None:
    mock_get_quote.return_value = (
        '{"results":{"AAPL":{"data":{"currentPrice":123.45},"meta":{}}},'
        '"summary":{"totalRequested":1,"totalReturned":1,"errors":[]}}'
    )

    response = await rest_client.get(
        "/quote",
        params=[
            ("symbols", "AAPL,MSFT"),
            ("symbols", "NVDA"),
            ("fields", "currentPrice,marketCap"),
            ("fields", "sector"),
            ("no_cache", "true"),
        ],
    )

    assert response.status_code == 200
    assert response.json()["summary"]["totalRequested"] == 1
    mock_get_quote.assert_awaited_once_with(
        ["AAPL", "MSFT", "NVDA"],
        fields=["currentPrice", "marketCap", "sector"],
        no_cache=True,
    )


@pytest.mark.asyncio
@patch("yfmcp.server.screen", new_callable=AsyncMock)
async def test_screen_post_accepts_json_body(mock_screen: AsyncMock, rest_client: httpx.AsyncClient) -> None:
    mock_screen.return_value = '{"finance":{"result":[{"id":"day_gainers"}],"error":null}}'

    response = await rest_client.post(
        "/screen",
        json={
            "query_type": "predefined",
            "query": "day_gainers",
            "count": 25,
            "sort_field": "percentchange",
        },
    )

    assert response.status_code == 200
    assert response.json()["finance"]["result"][0]["id"] == "day_gainers"
    mock_screen.assert_awaited_once_with(
        query="day_gainers",
        query_type="predefined",
        offset=None,
        size=None,
        count=25,
        sort_field="percentchange",
        sort_asc=None,
        user_id=None,
        user_id_type=None,
    )


@pytest.mark.asyncio
@patch("yfmcp.server.get_price_history", new_callable=AsyncMock)
async def test_price_history_chart_route_returns_image_bytes(
    mock_get_price_history: AsyncMock, rest_client: httpx.AsyncClient
) -> None:
    mock_get_price_history.return_value = ImageContent(
        type="image",
        data=base64.b64encode(b"webp-bytes").decode(),
        mimeType="image/webp",
    )

    response = await rest_client.get(
        "/price-history",
        params={"symbols": "AAPL", "chart_type": "price_volume"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.content == b"webp-bytes"


@pytest.mark.asyncio
async def test_quote_route_requires_symbols(rest_client: httpx.AsyncClient) -> None:
    response = await rest_client.get("/quote")

    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "INVALID_PARAMS"
    assert payload["details"]["parameter"] == "symbols"


@pytest.mark.asyncio
@patch("yfmcp.server._auth_provider", new=SharedSecretOAuthProvider("rest-secret"))
async def test_quote_route_requires_bearer_auth_when_mcp_auth_enabled(rest_client: httpx.AsyncClient) -> None:
    response = await rest_client.get("/quote", params={"symbols": "AAPL"})

    assert response.status_code == 401
    assert response.json()["error"] == "invalid_token"
    assert response.headers["www-authenticate"].startswith("Bearer ")


@pytest.mark.asyncio
@patch("yfmcp.server._auth_provider", new=SharedSecretOAuthProvider("rest-secret"))
@patch("yfmcp.server.get_quote", new_callable=AsyncMock)
async def test_quote_route_accepts_shared_secret_bearer_token(
    mock_get_quote: AsyncMock, rest_client: httpx.AsyncClient
) -> None:
    mock_get_quote.return_value = (
        '{"results":{"AAPL":{"data":{"currentPrice":123.45},"meta":{}}},'
        '"summary":{"totalRequested":1,"totalReturned":1,"errors":[]}}'
    )

    response = await rest_client.get(
        "/quote",
        params={"symbols": "AAPL"},
        headers={"Authorization": "Bearer rest-secret"},
    )

    assert response.status_code == 200
    assert response.json()["results"]["AAPL"]["data"]["currentPrice"] == 123.45
    mock_get_quote.assert_awaited_once_with(["AAPL"], fields=None, no_cache=False)
