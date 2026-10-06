"""End-to-end tests: the per-request ``session_guard`` re-gates a session.

Runs the real ``StreamableHTTPSessionManager`` behind a Starlette app (the
SDK's own runtime dependency), so what is exercised is the guard's
placement in the ASGI app: it must run on EVERY authenticated request —
including follow-up requests whose JSON-RPC frames the session manager
would otherwise forward to the already-running session task without any
re-validation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from mcp_dynamic_tool_registration.registry import ToolRegistry
from mcp_dynamic_tool_registration.server_factory import (
    build_streamable_http_asgi_app,
)

_TOKENS = {"token-alice": "alice", "token-bob": "bob"}

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

_INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}

_LIST_TOOLS = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

_CALL_TOOL = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "ping", "arguments": {}},
}


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
    assert _post(client, token, _INITIALIZED, session_id).status_code == 202
    return session_id


class TestGuardRunsOnEveryRequest:
    def test_established_session_is_re_gated_after_initialize(self):
        """The core regression: once a session is open, follow-up frames
        are forwarded to the session's persistent task WITHOUT any
        re-validation — the guard is the only per-request gate, so a
        guard that starts declining must 401 the next request on the
        open session."""
        declined: list[bool] = []

        def guard(token: str, request_context: Any) -> None:
            if declined:
                raise PermissionError("revoked")

        for client in _client(
            session_guard=guard, principal_of=lambda payload: payload["uid"]
        ):
            session_id = _open_session(client, "token-alice")

            # Sanity: the open session works while the guard passes.
            response = _post(client, "token-alice", _LIST_TOOLS, session_id)
            assert response.status_code == 200
            assert '"ping"' in response.text

            declined.append(True)

            response = _post(client, "token-alice", _LIST_TOOLS, session_id)
            assert response.status_code == 401
            assert response.json() == {"error": "Invalid or unauthorized token"}

            # Control-plane frames are gated too (the guard cannot tell
            # which JSON-RPC method a raw HTTP body carries).
            response = _post(client, "token-alice", _INITIALIZED, session_id)
            assert response.status_code == 401

    def test_a_tool_call_on_an_open_session_is_gated_before_dispatch(self):
        """The regression this guard exists for: a ``tools/call`` frame
        arriving on an already-open session must be rejected BEFORE it
        reaches the tool handler — the persistent task would otherwise
        execute the call with the context it inherited at session
        creation, with no re-validation anywhere."""
        declined: list[bool] = []
        handler_calls: list[Any] = []

        def ping(context: Any = None) -> dict[str, Any]:
            handler_calls.append(context)
            return {}

        def guard(token: str, request_context: Any) -> None:
            if declined:
                raise PermissionError("revoked")

        registry = ToolRegistry("demo")
        registry.add_tool(name="ping", description="Ping.", handler=ping)
        asgi_app, session_manager = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=_validate,
            session_guard=guard,
            principal_of=lambda payload: payload["uid"],
        )

        @asynccontextmanager
        async def lifespan(app: Starlette):
            async with session_manager.run():
                yield

        app = Starlette(
            routes=[Route("/mcp", endpoint=asgi_app, methods=["POST"])],
            lifespan=lifespan,
        )
        with TestClient(app) as client:
            session_id = _open_session(client, "token-alice")

            # Sanity: the open session executes the tool while the guard
            # passes, and the handler really runs.
            response = _post(client, "token-alice", _CALL_TOOL, session_id)
            assert response.status_code == 200
            assert '"isError":false' in response.text
            assert len(handler_calls) == 1

            declined.append(True)

            response = _post(client, "token-alice", _CALL_TOOL, session_id)
            assert response.status_code == 401
            assert response.json() == {"error": "Invalid or unauthorized token"}
            # The call never reached the tool — it was gated at the HTTP
            # boundary, not merely answered with an error result.
            assert len(handler_calls) == 1

    def test_guard_receives_the_raw_token_on_initialize_and_followups(self):
        calls: list[tuple[str, Any]] = []

        def guard(token: str, request_context: Any) -> None:
            calls.append((token, request_context))

        for client in _client(
            session_guard=guard, principal_of=lambda payload: payload["uid"]
        ):
            session_id = _open_session(client, "token-alice")
            assert _post(client, "token-alice", _LIST_TOOLS, session_id).status_code == 200

            assert [token for token, _ in calls] == [
                "token-alice",  # initialize
                "token-alice",  # notifications/initialized
                "token-alice",  # tools/list follow-up
            ]

    def test_declining_guard_on_initialize_gets_the_invalid_token_error_body(self):
        def guard(token: str, request_context: Any) -> None:
            raise PermissionError("nope")

        for client in _client(session_guard=guard):
            response = _post(client, "token-alice", _INITIALIZE)

            assert response.status_code == 401
            assert response.json() == {"error": "Invalid or unauthorized token"}

    def test_guard_receives_the_published_host_context_when_a_context_factory_exists(
        self,
    ):
        """With no ``context_factory`` there is no host context, so the
        guard is called with ``None`` for it; with one (the host's own
        wiring always has one), the guard receives the per-request host
        context already published on the library's contextvar."""
        seen: list[Any] = []

        def guard(token: str, request_context: Any) -> None:
            seen.append(request_context)

        for client in _client(session_guard=guard):
            response = _post(client, "token-alice", _INITIALIZE)
            assert response.status_code == 200

        assert len(seen) == 1
        assert seen[0] is None

        seen_contexts: list[Any] = []

        def guard2(token: str, request_context: Any) -> None:
            seen_contexts.append(request_context)

        registry = ToolRegistry("demo")
        asgi_app, session_manager = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=_validate,
            context_factory=lambda payload: {"uid": payload["uid"]},
            session_guard=guard2,
        )

        @asynccontextmanager
        async def lifespan(app: Starlette):
            async with session_manager.run():
                yield

        app = Starlette(
            routes=[Route("/mcp", endpoint=asgi_app, methods=["POST"])],
            lifespan=lifespan,
        )
        with TestClient(app) as client:
            assert _post(client, "token-alice", _INITIALIZE).status_code == 200

        assert seen_contexts == [{"uid": "alice"}]

    def test_guard_decline_does_not_leak_the_host_contextvar(self):
        import mcp_dynamic_tool_registration.server_factory as factory_module

        seen_contexts: list[Any] = []

        def guard(token: str, request_context: Any) -> None:
            seen_contexts.append(request_context)
            raise PermissionError("revoked")

        for client in _client(session_guard=guard):
            _post(client, "token-alice", _INITIALIZE)
            # After the declined request returns, the contextvar is back
            # to empty in this task (the guard path resets its set()).
            assert factory_module.current_request_context.get() is None


class TestGuardAbsentKeepsBehavior:
    def test_two_request_session_works_without_a_guard(self):
        for client in _client():
            session_id = _open_session(client, "token-alice")

            response = _post(client, "token-alice", _LIST_TOOLS, session_id)

            assert response.status_code == 200
            assert '"ping"' in response.text

    def test_missing_bearer_token_still_short_circuits_before_the_guard(self):
        calls: list[str] = []

        def guard(token: str, request_context: Any) -> None:
            calls.append(token)

        for client in _client(session_guard=guard):
            response = client.post(
                "/mcp",
                json=_INITIALIZE,
                headers={"Accept": "application/json, text/event-stream"},
            )

            assert response.status_code == 401
            assert calls == []
