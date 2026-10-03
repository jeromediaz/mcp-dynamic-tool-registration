"""Tests for build_streamable_http_asgi_app: the bearer-token-gated
Streamable HTTP ASGI app."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from mcp import types
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from mcp_dynamic_tool_registration.registry import ToolRegistry
from mcp_dynamic_tool_registration.server_factory import (
    build_streamable_http_asgi_app,
    current_request_context,
)


class TestBuildStreamableHttpAsgiApp:
    def test_returns_asgi_app_and_session_manager(self):
        registry = ToolRegistry("demo")
        asgi_app, session_manager = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=lambda t: {"uid": "u1"},
            context_factory=lambda payload: SimpleNamespace(uid=payload["uid"]),
        )
        assert callable(asgi_app)
        assert isinstance(session_manager, StreamableHTTPSessionManager)

    def test_rejects_missing_token(self):
        registry = ToolRegistry("demo")
        asgi_app, _ = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=lambda t: {"uid": "u1"},
            context_factory=lambda payload: SimpleNamespace(uid=payload["uid"]),
        )

        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.disconnect"}

        scope = {"type": "http", "headers": []}
        asyncio.run(asgi_app(scope, receive, send))

        assert sent[0]["status"] == 401
        body = json.loads(sent[1]["body"])
        assert "Missing bearer token" in body["error"]

    def test_rejects_invalid_token(self):
        registry = ToolRegistry("demo")

        def token_validator(token: str) -> dict:
            raise ValueError("bad token")

        asgi_app, _ = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=token_validator,
            context_factory=lambda payload: SimpleNamespace(uid=payload["uid"]),
        )

        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.disconnect"}

        scope = {"type": "http", "headers": [(b"authorization", b"Bearer bad")]}
        asyncio.run(asgi_app(scope, receive, send))

        assert sent[0]["status"] == 401

    def test_valid_token_delegates_to_session_manager(self):
        registry = ToolRegistry("demo")
        calls = []

        async def fake_handle_request(scope, receive, send):
            calls.append(scope)

        asgi_app, session_manager = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=lambda t: {"uid": "u1"},
            context_factory=lambda payload: SimpleNamespace(uid=payload["uid"]),
        )
        session_manager.handle_request = fake_handle_request  # type: ignore[method-assign]

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {"type": "http", "headers": [(b"authorization", b"Bearer good")]}
        asyncio.run(asgi_app(scope, receive, send))

        assert len(calls) == 1

    def test_valid_token_sets_request_context_for_duration_of_handle_request(self):
        """The per-request context (built from the validated token) must be
        readable via current_request_context while handle_request runs, and
        reset afterward — this is what lets the call-tool handler inject
        `context` per call instead of a static registration-time default."""
        registry = ToolRegistry("demo")
        observed = {}

        async def fake_handle_request(scope, receive, send):
            observed["during"] = current_request_context.get()

        asgi_app, session_manager = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=lambda t: {"uid": "u-context-test"},
            context_factory=lambda payload: SimpleNamespace(uid=payload["uid"]),
        )
        session_manager.handle_request = fake_handle_request  # type: ignore[method-assign]

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {"type": "http", "headers": [(b"authorization", b"Bearer good")]}
        assert current_request_context.get() is None

        async def run():
            await asgi_app(scope, receive, send)
            # "after" assertion runs inside the same coroutine: the var must
            # be reset right after the app returns, so it never leaks to the
            # next request.
            observed["after"] = current_request_context.get()

        asyncio.run(run())

        assert observed["during"] is not None
        assert observed["during"].uid == "u-context-test"
        assert observed["after"] is None

    def test_non_http_scope_is_noop(self):
        registry = ToolRegistry("demo")
        asgi_app, _ = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=lambda t: {"uid": "u1"},
            context_factory=lambda payload: SimpleNamespace(uid=payload["uid"]),
        )

        called = False

        async def send(message):
            nonlocal called
            called = True

        async def receive():
            return {"type": "websocket.disconnect"}

        scope = {"type": "websocket", "headers": []}
        asyncio.run(asgi_app(scope, receive, send))
        assert called is False


class TestAsgiAppSeams:
    """Tests for the host-seam parameters: ``context_factory=None``,
    ``request_hook``, and how they interact with the per-request
    ``current_request_context``."""

    def test_context_factory_none_keeps_var_unset_but_still_handles(self):
        """With no ``context_factory`` the app publishes nothing on
        ``current_request_context``, yet still delegates the request to the
        session manager (the server is built without context injection)."""
        registry = ToolRegistry("demo")
        observed = {}

        async def fake_handle_request(scope, receive, send):
            observed["during"] = current_request_context.get()

        asgi_app, session_manager = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=lambda t: {"uid": "u1"},
        )
        session_manager.handle_request = fake_handle_request  # type: ignore[method-assign]

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {
            "type": "http",
            "method": "POST",
            "headers": [(b"authorization", b"Bearer good")],
        }

        async def run():
            await asgi_app(scope, receive, send)
            # Assert "after" inside the same coroutine, right after the app
            # returns, so a leaked context var would be caught here.
            observed["after"] = current_request_context.get()

        asyncio.run(run())

        assert observed["during"] is None
        assert observed["after"] is None

    def test_request_hook_called_with_name_session_and_method(self):
        registry = ToolRegistry("demo")
        calls = []

        async def fake_handle_request(scope, receive, send):
            pass

        asgi_app, session_manager = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=lambda t: {"uid": "u1"},
            context_factory=lambda payload: SimpleNamespace(uid=payload["uid"]),
            request_hook=lambda name, session_id, method: calls.append(
                (name, session_id, method)
            ),
        )
        session_manager.handle_request = fake_handle_request  # type: ignore[method-assign]

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {
            "type": "http",
            "method": "POST",
            "headers": [
                (b"authorization", b"Bearer good"),
                (b"mcp-session-id", b"s-1"),
            ],
        }
        asyncio.run(asgi_app(scope, receive, send))

        assert calls == [("demo", "s-1", "POST")]

    def test_request_hook_not_called_on_401(self):
        """The hook runs *after* token validation, so an unauthenticated
        request never reaches it — host instrumentation stays behind the
        auth gate."""
        registry = ToolRegistry("demo")
        calls = []

        async def fake_handle_request(scope, receive, send):
            pass

        asgi_app, session_manager = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=lambda t: {"uid": "u1"},
            context_factory=lambda payload: SimpleNamespace(uid=payload["uid"]),
            request_hook=lambda name, session_id, method: calls.append(
                (name, session_id, method)
            ),
        )
        session_manager.handle_request = fake_handle_request  # type: ignore[method-assign]

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {"type": "http", "method": "POST", "headers": []}
        asyncio.run(asgi_app(scope, receive, send))

        assert calls == []

    def test_request_hook_raising_does_not_break_the_request(self):
        """A failing ``request_hook`` is logged and swallowed: the request
        still reaches ``handle_request`` with its context published, and the
        context var is still reset afterward."""
        registry = ToolRegistry("demo")
        events = []

        def broken_hook(name, session_id, method):
            events.append((name, session_id, method))
            raise RuntimeError("hook exploded")

        async def fake_handle_request(scope, receive, send):
            events.append(current_request_context.get())

        asgi_app, session_manager = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=lambda t: {"uid": "u-hook"},
            context_factory=lambda payload: SimpleNamespace(uid=payload["uid"]),
            request_hook=broken_hook,
        )
        session_manager.handle_request = fake_handle_request  # type: ignore[method-assign]

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {
            "type": "http",
            "method": "POST",
            "headers": [
                (b"authorization", b"Bearer good"),
                (b"mcp-session-id", b"s-2"),
            ],
        }

        observed = {}

        async def run():
            await asgi_app(scope, receive, send)
            observed["after"] = current_request_context.get()

        asyncio.run(run())

        assert events[0] == ("demo", "s-2", "POST")
        # handle_request ran and saw the per-request context.
        assert events[1] is not None
        assert events[1].uid == "u-hook"
        # The var was still reset by the finally, despite the hook failing.
        assert observed["after"] is None

    def test_context_flows_from_asgi_to_handler_end_to_end(self):
        """Full-pipeline guard against two out-of-sync ContextVars: a valid
        bearer request through the ASGI app must reach the tool handler with
        the ``context_factory`` output (built from the validated token) as
        its ``context`` kwarg. The ``with context:`` frame in the dispatch is
        what makes it readable inside the handler."""
        registry = ToolRegistry("demo")
        captured = {}

        def whoami(context=None):
            captured["context"] = context
            return {"ok": True}

        # The handler declares neither an mcp_session parameter nor uses
        # elicitation, so it runs without a live request context and the
        # dispatch never touches mcp's own request_ctx.
        registry.add_tool(name="whoami", description="", handler=whoami)

        asgi_app, session_manager = build_streamable_http_asgi_app(
            "demo",
            registry,
            token_validator=lambda t: {"uid": "u-e2e"},
            context_factory=lambda payload: SimpleNamespace(uid=payload["uid"]),
        )

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        async def fake_handle_request(scope, receive, send):
            call_handler = session_manager.app.request_handlers[types.CallToolRequest]
            await call_handler(
                types.CallToolRequest(
                    method="tools/call",
                    params=types.CallToolRequestParams(name="whoami", arguments={}),
                )
            )

        session_manager.handle_request = fake_handle_request  # type: ignore[method-assign]

        scope = {
            "type": "http",
            "method": "POST",
            "headers": [(b"authorization", b"Bearer good")],
        }
        asyncio.run(asgi_app(scope, receive, send))

        assert captured["context"] is not None
        assert captured["context"].uid == "u-e2e"
