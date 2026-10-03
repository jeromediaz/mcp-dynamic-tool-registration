"""End-to-end tests: an MCP session is bound to the caller that created it.

Runs the real ``StreamableHTTPSessionManager`` behind a Starlette app (the
SDK's own runtime dependency), so the SDK's session-ownership check is what
is being exercised.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from mcp_dynamic_tool_registration.registry import ToolRegistry
from mcp_dynamic_tool_registration.server_factory import (
    build_streamable_http_asgi_app,
)

_TOKENS = {"token-alice": "alice", "token-alice-2": "alice", "token-bob": "bob"}

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}

_LIST_TOOLS = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


def _validate(token: str) -> dict[str, str]:
    if token not in _TOKENS:
        raise PermissionError("unknown token")
    return {"uid": _TOKENS[token]}


def _client(**app_kwargs: Any) -> Iterator[TestClient]:
    registry = ToolRegistry("demo")
    registry.add_tool(name="ping", description="Ping.", handler=lambda context=None: {})
    asgi_app, session_manager = build_streamable_http_asgi_app(
        "demo", registry, token_validator=_validate, **app_kwargs
    )

    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with session_manager.run():
            yield

    app = Starlette(
        routes=[Route("/mcp", endpoint=asgi_app, methods=["GET", "POST", "DELETE"])],
        lifespan=lifespan,
    )
    with TestClient(app) as client:
        yield client


def _post(client: TestClient, token: str, body: dict, session_id: str | None = None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if session_id is not None:
        headers["mcp-session-id"] = session_id
    return client.post("/mcp", json=body, headers=headers)


def _open_session(client: TestClient, token: str) -> str:
    response = _post(client, token, _INITIALIZE)
    assert response.status_code == 200
    session_id = response.headers["mcp-session-id"]
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert _post(client, token, initialized, session_id).status_code == 202
    return session_id


@pytest.fixture
def default_client() -> Iterator[TestClient]:
    yield from _client()


@pytest.fixture
def principal_client() -> Iterator[TestClient]:
    yield from _client(principal_of=lambda payload: payload["uid"])


class TestDefaultBindingIsPerToken:
    def test_creator_can_use_its_session(self, default_client):
        session_id = _open_session(default_client, "token-alice")

        response = _post(default_client, "token-alice", _LIST_TOOLS, session_id)

        assert response.status_code == 200
        assert '"ping"' in response.text

    def test_other_caller_gets_session_not_found(self, default_client):
        session_id = _open_session(default_client, "token-alice")

        response = _post(default_client, "token-bob", _LIST_TOOLS, session_id)

        assert response.status_code == 404
        assert "Session not found" in response.text

    def test_other_token_of_same_user_is_rejected_by_default(self, default_client):
        session_id = _open_session(default_client, "token-alice")

        response = _post(default_client, "token-alice-2", _LIST_TOOLS, session_id)

        assert response.status_code == 404


class TestPrincipalOfBindsToTheCaller:
    def test_other_token_of_same_principal_is_accepted(self, principal_client):
        session_id = _open_session(principal_client, "token-alice")

        response = _post(principal_client, "token-alice-2", _LIST_TOOLS, session_id)

        assert response.status_code == 200

    def test_other_principal_gets_session_not_found(self, principal_client):
        session_id = _open_session(principal_client, "token-alice")

        response = _post(principal_client, "token-bob", _LIST_TOOLS, session_id)

        assert response.status_code == 404

    def test_principal_of_raising_is_unauthorized(self):
        def broken(payload: Any) -> str:
            raise KeyError("uid")

        for client in _client(principal_of=broken):
            response = _post(client, "token-alice", _INITIALIZE)
            assert response.status_code == 401
            assert response.json() == {"error": "Invalid or unauthorized token"}
