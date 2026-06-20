"""Minimal OAuth 2.1 authorization server gated by a single shared secret.

This lets the server be added as a Claude (or any MCP client) connector via the
standard OAuth discovery/consent flow, without a real user/account system: the
"consent screen" is a password form that checks the request's secret against
MCP_AUTH_SECRET. Dynamic client registration is always allowed; authorization
codes, access tokens, and refresh tokens are kept in memory only, so they reset
on restart and don't survive across multiple server replicas.
"""

from __future__ import annotations

import secrets
import time

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.provider import AuthorizationCode
from mcp.server.auth.provider import AuthorizationParams
from mcp.server.auth.provider import AuthorizeError
from mcp.server.auth.provider import OAuthAuthorizationServerProvider
from mcp.server.auth.provider import RefreshToken
from mcp.server.auth.provider import construct_redirect_uri
from mcp.shared.auth import OAuthClientInformationFull
from mcp.shared.auth import OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.responses import RedirectResponse

ACCESS_TOKEN_TTL_SECONDS = 3600
AUTHORIZATION_CODE_TTL_SECONDS = 300

LOGIN_FORM = """
<!doctype html>
<title>yfinance-mcp authorization</title>
<form method="post">
  <p>Enter the access secret to connect to yfinance-mcp.</p>
  <input type="password" name="secret" placeholder="Secret" autofocus required>
  <button type="submit">Authorize</button>
  {error}
</form>
"""


class SharedSecretOAuthProvider(OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]):
    def __init__(self, secret: str) -> None:
        self._secret = secret
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._pending_authorizations: dict[str, tuple[OAuthClientInformationFull, AuthorizationParams]] = {}
        self._authorization_codes: dict[str, AuthorizationCode] = {}
        self._access_tokens: dict[str, AccessToken] = {}
        self._refresh_tokens: dict[str, RefreshToken] = {}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        assert client_info.client_id is not None, "registration always assigns a client_id"
        self._clients[client_info.client_id] = client_info

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        request_id = secrets.token_urlsafe(32)
        self._pending_authorizations[request_id] = (client, params)
        return f"/login?request_id={request_id}"

    def _complete_authorization(self, request_id: str) -> str | None:
        pending = self._pending_authorizations.pop(request_id, None)
        if pending is None:
            return None
        client, params = pending
        assert client.client_id is not None, "registered clients always have a client_id"
        code = secrets.token_urlsafe(32)
        self._authorization_codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + AUTHORIZATION_CODE_TTL_SECONDS,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        code = self._authorization_codes.get(authorization_code)
        if code is None or code.client_id != client.client_id:
            return None
        return code

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        assert client.client_id is not None, "registered clients always have a client_id"
        self._authorization_codes.pop(authorization_code.code, None)
        return self._issue_tokens(client.client_id, authorization_code.scopes, authorization_code.resource)

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        token = self._refresh_tokens.get(refresh_token)
        if token is None or token.client_id != client.client_id:
            return None
        return token

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        assert client.client_id is not None, "registered clients always have a client_id"
        self._refresh_tokens.pop(refresh_token.token, None)
        return self._issue_tokens(client.client_id, scopes)

    async def load_access_token(self, token: str) -> AccessToken | None:
        access_token = self._access_tokens.get(token)
        if access_token is None:
            return None
        if access_token.expires_at and access_token.expires_at < time.time():
            self._access_tokens.pop(token, None)
            return None
        return access_token

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, AccessToken):
            self._access_tokens.pop(token.token, None)
        else:
            self._refresh_tokens.pop(token.token, None)

    def _issue_tokens(self, client_id: str, scopes: list[str], resource: str | None = None) -> OAuthToken:
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)
        expires_at = int(time.time() + ACCESS_TOKEN_TTL_SECONDS)
        self._access_tokens[access_token] = AccessToken(
            token=access_token, client_id=client_id, scopes=scopes, expires_at=expires_at, resource=resource
        )
        self._refresh_tokens[refresh_token] = RefreshToken(token=refresh_token, client_id=client_id, scopes=scopes)
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL_SECONDS,
            refresh_token=refresh_token,
            scope=" ".join(scopes) if scopes else None,
        )

    async def handle_login(self, request: Request) -> HTMLResponse | RedirectResponse:
        request_id = request.query_params.get("request_id") or (await request.form()).get("request_id")
        if not isinstance(request_id, str) or request_id not in self._pending_authorizations:
            raise AuthorizeError(error="invalid_request", error_description="Unknown or expired authorization request")

        if request.method == "GET":
            return HTMLResponse(LOGIN_FORM.format(error=""))

        form = await request.form()
        secret = form.get("secret")
        if secret != self._secret:
            return HTMLResponse(LOGIN_FORM.format(error="<p>Incorrect secret.</p>"), status_code=401)

        redirect_url = self._complete_authorization(request_id)
        if redirect_url is None:
            raise AuthorizeError(error="invalid_request", error_description="Unknown or expired authorization request")
        return RedirectResponse(url=redirect_url, status_code=302)
