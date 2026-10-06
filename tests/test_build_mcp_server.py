"""Tests for build_mcp_server: registry-backed tool listing and dispatch,
argument coercion/validation, and the host hooks."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

from mcp import types
from mcp.server.lowlevel import Server

from mcp_dynamic_tool_registration.registry import ToolRegistry, ToolSpec
from mcp_dynamic_tool_registration.server_factory import (
    build_mcp_server,
    current_request_context,
    default_error_handler,
)


def _call(server: Server, name: str, arguments: dict[str, Any]) -> Any:
    handler = server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    return asyncio.run(handler(request)).root


class _HTTPError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _raise(exc: Exception) -> Any:
    raise exc


class TestBuildMcpServer:
    def test_returns_lowlevel_server(self):
        registry = ToolRegistry("demo")
        server = build_mcp_server("demo", registry)
        assert isinstance(server, Server)

    def test_list_tools_reflects_registry(self):
        registry = ToolRegistry("demo")
        registry.add_tool(
            name="ping",
            description="Ping.",
            handler=lambda: {"pong": True},
            annotations={"readOnlyHint": True},
        )
        server = build_mcp_server("demo", registry)

        list_tools_handler = server.request_handlers[types.ListToolsRequest]
        result = asyncio.run(
            list_tools_handler(types.ListToolsRequest(method="tools/list"))
        )
        tools = result.root.tools
        assert len(tools) == 1
        assert tools[0].name == "ping"
        assert tools[0].annotations.readOnlyHint is True

    def test_call_tool_dispatches_to_handler(self):
        registry = ToolRegistry("demo")
        registry.add_tool(
            name="echo",
            description="Echo.",
            handler=lambda text, context=None: {"text": text},
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="echo", arguments={"text": "hi"}),
        )
        result = asyncio.run(call_tool_handler(request))
        assert result.root.structuredContent == {"text": "hi"}
        assert result.root.isError is False

    def test_call_tool_passes_current_request_context(self):
        """_call_tool must inject the current per-request context (set via
        current_request_context, not a registration-time static default) as
        a `context` kwarg on every call."""
        registry = ToolRegistry("demo")
        captured = {}

        def handler(context=None):
            captured["context"] = context
            return {"ok": True}

        registry.add_tool(name="whoami", description="", handler=handler)
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="whoami", arguments={}),
        )

        sentinel_context = MagicMock()  # supports the `with context:` protocol
        token = current_request_context.set(sentinel_context)
        try:
            asyncio.run(call_tool_handler(request))
        finally:
            current_request_context.reset(token)

        assert captured["context"] is sentinel_context

    def test_call_tool_context_defaults_to_none_outside_a_request(self):
        registry = ToolRegistry("demo")
        captured = {}

        def handler(context=None):
            captured["context"] = context
            return {"ok": True}

        registry.add_tool(name="whoami", description="", handler=handler)
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="whoami", arguments={}),
        )
        asyncio.run(call_tool_handler(request))

        assert captured["context"] is None

    def test_call_tool_unknown_tool_returns_error_result(self):
        registry = ToolRegistry("demo")
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="nope", arguments={}),
        )
        result = asyncio.run(call_tool_handler(request))
        assert result.root.isError is True

    def test_call_tool_unexpected_argument_returns_error_without_calling_handler(self):
        """A bad kwarg (e.g. a typo'd field name) must be rejected as a
        clean isError tool result against input_schema, never reach the
        handler as a raw TypeError, which the SDK would log as a server
        fault although it is just a malformed tool call."""
        from pydantic import BaseModel

        class EchoInput(BaseModel):
            text: str

        registry = ToolRegistry("demo")
        called = MagicMock()
        registry.add_tool(
            name="echo",
            description="Echo.",
            input_schema=EchoInput,
            handler=lambda text, context=None: called() or {"text": text},
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="echo", arguments={"text": "hi", "markdown": "hi"}
            ),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is True
        assert "markdown" in result.root.content[0].text
        called.assert_not_called()

    def test_call_tool_missing_required_argument_returns_error_result(self):
        from pydantic import BaseModel

        class EchoInput(BaseModel):
            text: str

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="echo",
            description="Echo.",
            input_schema=EchoInput,
            handler=lambda text, context=None: {"text": text},
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="echo", arguments={}),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is True

    def test_call_tool_valid_arguments_still_dispatch_normally(self):
        from pydantic import BaseModel

        class EchoInput(BaseModel):
            text: str

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="echo",
            description="Echo.",
            input_schema=EchoInput,
            handler=lambda text, context=None: {"text": text},
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="echo", arguments={"text": "hi"}),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is False
        assert result.root.structuredContent == {"text": "hi"}

    def test_call_tool_injects_mcp_session_only_when_declared(self):
        """Destructive tools declare an `mcp_session` parameter — the
        dispatch must inject the current request's ServerSession
        (via mcp's own request_ctx, the same mechanism Server._handle_request
        uses in production) only for handlers that actually declare it,
        the same "only pass what's recognized" filtering
        _bind_params_as_defaults applies to registration-time kwargs."""
        from mcp.server.lowlevel.server import RequestContext, request_ctx

        registry = ToolRegistry("demo")
        captured = {}

        def with_session(context=None, mcp_session=None):
            captured["with_session"] = mcp_session
            return {"ok": True}

        def without_session(context=None):
            captured["without_session_called"] = True
            return {"ok": True}

        registry.add_tool(name="with_session", description="", handler=with_session)
        registry.add_tool(
            name="without_session", description="", handler=without_session
        )
        server = build_mcp_server("demo", registry)

        sentinel_session = MagicMock()
        rc_token = request_ctx.set(
            RequestContext(
                request_id="req-1",
                meta=None,
                session=sentinel_session,
                lifespan_context=None,
            )
        )
        try:
            call_tool_handler = server.request_handlers[types.CallToolRequest]
            asyncio.run(
                call_tool_handler(
                    types.CallToolRequest(
                        method="tools/call",
                        params=types.CallToolRequestParams(
                            name="with_session", arguments={}
                        ),
                    )
                )
            )
            asyncio.run(
                call_tool_handler(
                    types.CallToolRequest(
                        method="tools/call",
                        params=types.CallToolRequestParams(
                            name="without_session", arguments={}
                        ),
                    )
                )
            )
        finally:
            request_ctx.reset(rc_token)

        assert captured["with_session"] is sentinel_session
        assert captured["without_session_called"] is True

    def test_call_tool_http_exception_returns_error_result_not_raises(self):
        """A handler raising a 4xx (e.g. an HTTPException-style "not found"
        or "invalid key") must come back as an isError CallToolResult the
        client can read, never propagate to the SDK's request handler, which
        would log it as a server fault."""

        class FakeHTTPException(Exception):
            def __init__(self, status_code: int, detail: str):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="resolve_note_key",
            description="",
            handler=lambda key, context=None: (_ for _ in ()).throw(
                FakeHTTPException(422, "'foo' is not a valid note key")
            ),
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="resolve_note_key", arguments={"key": "foo"}
            ),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is True
        assert "not a valid note key" in result.root.content[0].text

    def test_call_tool_elicitation_not_supported_returns_error_result(self):
        """A destructive tool whose client didn't negotiate elicitation
        raises ElicitationNotSupportedError — a caller/capability error that
        must come back as an isError result, not reach the SDK's
        exception logging."""
        from mcp_dynamic_tool_registration.elicitation import (
            ElicitationNotSupportedError,
        )

        registry = ToolRegistry("demo")

        def handler(context=None, mcp_session=None):
            raise ElicitationNotSupportedError(
                "Connected MCP client does not support elicitation"
            )

        registry.add_tool(name="delete_note", description="", handler=handler)
        server = build_mcp_server("demo", registry)

        from mcp.server.lowlevel.server import RequestContext, request_ctx

        sentinel_session = MagicMock()
        rc_token = request_ctx.set(
            RequestContext(
                request_id="req-1",
                meta=None,
                session=sentinel_session,
                lifespan_context=None,
            )
        )
        try:
            call_tool_handler = server.request_handlers[types.CallToolRequest]
            request = types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="delete_note", arguments={"note_id": "abc"}
                ),
            )
            result = asyncio.run(call_tool_handler(request))
        finally:
            request_ctx.reset(rc_token)

        assert result.root.isError is True
        assert "elicitation" in result.root.content[0].text.lower()

    def test_call_tool_declined_error_returns_error_result(self):
        """A user declining a destructive confirmation (DeclinedError) is a
        normal outcome, not a fault — isError result, not a logged exception."""
        from mcp_dynamic_tool_registration.elicitation import DeclinedError

        registry = ToolRegistry("demo")

        def handler(context=None, mcp_session=None):
            raise DeclinedError("User declined confirmation to delete note")

        registry.add_tool(name="delete_note", description="", handler=handler)
        server = build_mcp_server("demo", registry)

        from mcp.server.lowlevel.server import RequestContext, request_ctx

        sentinel_session = MagicMock()
        rc_token = request_ctx.set(
            RequestContext(
                request_id="req-1",
                meta=None,
                session=sentinel_session,
                lifespan_context=None,
            )
        )
        try:
            call_tool_handler = server.request_handlers[types.CallToolRequest]
            request = types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="delete_note", arguments={"note_id": "abc"}
                ),
            )
            result = asyncio.run(call_tool_handler(request))
        finally:
            request_ctx.reset(rc_token)

        assert result.root.isError is True
        assert "declined" in result.root.content[0].text.lower()

    def test_call_tool_5xx_and_plain_exceptions_become_error_results(self):
        """A real server fault (5xx or an unhandled exception) is NOT
        swallowed by the default error handler — it propagates to the mcp SDK's own
        exception handler, which converts it into an ``isError``
        ``CallToolResult``.  This test verifies that pipeline: the 5xx
        message must appear in the error result text."""

        class FakeServerException(Exception):
            def __init__(self, status_code: int, detail: str):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="boom",
            description="",
            handler=lambda context=None: (_ for _ in ()).throw(
                FakeServerException(500, "internal fault")
            ),
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="boom", arguments={}),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is True
        assert "internal fault" in result.root.content[0].text


class TestCallToolCoercion:
    """Integration tests: _call_tool must coerce double-encoded arguments
    before both validation and the handler call."""

    def test_double_encoded_list_argument_is_coerced(self):
        from pydantic import BaseModel

        class NoteInput(BaseModel):
            title: str
            tags: list[str] | None = None

        captured = {}

        def handler(context=None, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="create_note",
            description="",
            input_schema=NoteInput,
            handler=handler,
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="create_note",
                arguments={
                    "title": "test",
                    "tags": '["a", "b", "c"]',
                },
            ),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is False
        assert captured["tags"] == ["a", "b", "c"]

    def test_double_encoded_dict_list_argument_is_coerced(self):
        from pydantic import BaseModel

        class RelationEntry(BaseModel):
            predicate: str
            target_type: str
            target_id: str

        class NoteInput(BaseModel):
            title: str
            relations: list[RelationEntry] | None = None

        captured = {}

        def handler(context=None, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="create_note",
            description="",
            input_schema=NoteInput,
            handler=handler,
        )
        server = build_mcp_server("demo", registry)

        relations_json = json.dumps(
            [
                {
                    "predicate": "references",
                    "target_type": "note",
                    "target_id": "abc123",
                }
            ]
        )
        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="create_note",
                arguments={"title": "test", "relations": relations_json},
            ),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is False
        assert captured["relations"] == [
            {
                "predicate": "references",
                "target_type": "note",
                "target_id": "abc123",
            }
        ]

    def test_properly_encoded_arguments_pass_through_unchanged(self):
        from pydantic import BaseModel

        class NoteInput(BaseModel):
            title: str
            tags: list[str] | None = None

        captured = {}

        def handler(context=None, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="create_note",
            description="",
            input_schema=NoteInput,
            handler=handler,
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="create_note",
                arguments={"title": "test", "tags": ["x", "y"]},
            ),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is False
        assert captured["tags"] == ["x", "y"]

    def test_genuinely_invalid_arguments_still_rejected(self):
        from pydantic import BaseModel

        class NoteInput(BaseModel):
            title: str
            tags: list[str] | None = None

        called = MagicMock()
        registry = ToolRegistry("demo")
        registry.add_tool(
            name="create_note",
            description="",
            input_schema=NoteInput,
            handler=lambda **kw: called() or {"ok": True},
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="create_note",
                arguments={"title": "test", "tags": "not-a-json-array"},
            ),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is True
        called.assert_not_called()

    def test_scalar_body_starting_with_brace_reaches_handler_verbatim(self):
        from pydantic import BaseModel

        class NoteInput(BaseModel):
            title: str
            body: str = ""

        captured = {}

        def handler(context=None, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="create_note",
            description="",
            input_schema=NoteInput,
            handler=handler,
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="create_note",
                arguments={"title": "t", "body": '{"a": 1}'},
            ),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is False
        assert captured["body"] == '{"a": 1}'

    def test_nested_scalar_field_not_double_decoded(self):
        from pydantic import BaseModel

        class NoteInput(BaseModel):
            title: str
            tags: list[str] | None = None

        captured = {}

        def handler(context=None, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="create_note",
            description="",
            input_schema=NoteInput,
            handler=handler,
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="create_note",
                arguments={"title": '["a"]', "tags": ["x", "y"]},
            ),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is False
        assert captured["title"] == '["a"]'

    def test_double_encoded_list_under_container_field_still_coerced(self):
        from pydantic import BaseModel

        class NoteInput(BaseModel):
            title: str
            tags: list[str] | None = None

        captured = {}

        def handler(context=None, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="create_note",
            description="",
            input_schema=NoteInput,
            handler=handler,
        )
        server = build_mcp_server("demo", registry)

        call_tool_handler = server.request_handlers[types.CallToolRequest]
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="create_note",
                arguments={"title": "test", "tags": '["a", "b", "c"]'},
            ),
        )
        result = asyncio.run(call_tool_handler(request))

        assert result.root.isError is False
        assert captured["tags"] == ["a", "b", "c"]


class TestHooks:
    def test_inject_context_false_calls_handler_without_context(self):
        registry = ToolRegistry("demo")
        registry.add_tool(
            name="add", description="", handler=lambda a, b: {"sum": a + b}
        )
        server = build_mcp_server("demo", registry, inject_context=False)

        result = _call(server, "add", {"a": 1, "b": 2})

        assert result.isError is False
        assert result.structuredContent == {"sum": 3}

    def test_audit_hook_receives_call_and_its_result_is_returned(self):
        registry = ToolRegistry("demo")
        handler = MagicMock(return_value={"direct": True})
        registry.add_tool(name="t", description="", handler=handler)
        recorded: dict[str, Any] = {}

        async def audit_hook(**kwargs: Any) -> Any:
            recorded.update(kwargs)
            return {"audited": True}

        server = build_mcp_server("demo", registry, audit_hook=audit_hook)
        result = _call(server, "t", {"x": 1})

        assert set(recorded) == {
            "tool_name",
            "spec",
            "context",
            "arguments",
            "extra_handler_kwargs",
        }
        assert recorded["tool_name"] == "t"
        assert isinstance(recorded["spec"], ToolSpec)
        assert recorded["spec"] is registry.get_tool("t")
        assert recorded["context"] is None
        assert recorded["arguments"] == {"x": 1}
        assert recorded["extra_handler_kwargs"] == {}
        assert result.structuredContent == {"audited": True}
        handler.assert_not_called()

    def test_audit_hook_4xx_becomes_usage_error(self):
        registry = ToolRegistry("demo")
        registry.add_tool(name="t", description="", handler=lambda context=None: {})

        async def audit_hook(**kwargs: Any) -> Any:
            raise _HTTPError(404, "gone")

        server = build_mcp_server("demo", registry, audit_hook=audit_hook)
        result = _call(server, "t", {})

        assert result.isError is True
        assert result.content[0].text == "Tool 't' error: gone"

    def test_error_handler_none_leaves_4xx_to_the_sdk(self):
        registry = ToolRegistry("demo")
        registry.add_tool(
            name="t",
            description="",
            handler=lambda context=None: _raise(_HTTPError(404, "gone")),
        )
        server = build_mcp_server("demo", registry, error_handler=None)

        result = _call(server, "t", {})

        # The SDK still reports an error, but with its own plain message
        # rather than this library's "Tool '<name>' error:" usage format.
        assert result.isError is True
        assert not result.content[0].text.startswith("Tool '")
        assert "gone" in result.content[0].text

    def test_custom_error_handler_result_is_used(self):
        registry = ToolRegistry("demo")
        registry.add_tool(
            name="t",
            description="",
            handler=lambda context=None: _raise(RuntimeError("boom")),
        )
        custom = types.CallToolResult(
            content=[types.TextContent(type="text", text="handled")], isError=True
        )
        seen: list[tuple[str, Exception]] = []

        def error_handler(tool_name: str, exc: Exception) -> types.CallToolResult:
            seen.append((tool_name, exc))
            return custom

        server = build_mcp_server("demo", registry, error_handler=error_handler)
        result = _call(server, "t", {})

        assert result.content[0].text == "handled"
        assert seen[0][0] == "t"
        assert isinstance(seen[0][1], RuntimeError)

    def test_coerce_args_false_rejects_double_encoded_list(self):
        from pydantic import BaseModel

        class TagsInput(BaseModel):
            tags: list[str]

        registry = ToolRegistry("demo")
        handler = MagicMock(return_value={})
        registry.add_tool(
            name="t", description="", input_schema=TagsInput, handler=handler
        )
        server = build_mcp_server("demo", registry, coerce_args=False)

        result = _call(server, "t", {"tags": json.dumps(["a", "b"])})

        assert result.isError is True
        assert result.content[0].text.startswith("Invalid arguments for tool 't'")
        handler.assert_not_called()

    def test_non_context_manager_context_is_passed_through(self):
        registry = ToolRegistry("demo")
        captured: dict[str, Any] = {}

        def handler(context=None):
            captured["context"] = context
            return {}

        registry.add_tool(name="t", description="", handler=handler)
        server = build_mcp_server("demo", registry)
        plain_context = object()

        token = current_request_context.set(plain_context)
        try:
            _call(server, "t", {})
        finally:
            current_request_context.reset(token)

        assert captured["context"] is plain_context

    def test_context_manager_context_is_entered_around_the_call(self):
        registry = ToolRegistry("demo")
        events: list[str] = []

        class Ctx:
            def __enter__(self) -> Ctx:
                events.append("enter")
                return self

            def __exit__(self, *exc_info: object) -> None:
                events.append("exit")

        def handler(context=None):
            events.append("call")
            return {}

        registry.add_tool(name="t", description="", handler=handler)
        server = build_mcp_server("demo", registry)

        token = current_request_context.set(Ctx())
        try:
            _call(server, "t", {})
        finally:
            current_request_context.reset(token)

        assert events == ["enter", "call", "exit"]


class TestArgumentBounds:
    """_call_tool must reject out-of-bounds arguments (sizes, not shapes)
    as a clean isError result, before the audit hook or handler run."""

    @staticmethod
    def _echo_server(**limit_kwargs: Any) -> Server:
        from pydantic import BaseModel

        from mcp_dynamic_tool_registration.server_factory import ArgumentLimits

        class EchoInput(BaseModel):
            text: str

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="echo",
            description="Echo.",
            input_schema=EchoInput,
            handler=lambda text, context=None: {"text": text},
        )
        return build_mcp_server(
            "demo", registry, argument_limits=ArgumentLimits(**limit_kwargs)
        )

    def test_oversized_string_is_rejected_without_calling_handler(self):
        from pydantic import BaseModel

        from mcp_dynamic_tool_registration.server_factory import ArgumentLimits

        class EchoInput(BaseModel):
            text: str

        called = MagicMock()
        registry = ToolRegistry("demo")
        registry.add_tool(
            name="echo",
            description="Echo.",
            input_schema=EchoInput,
            handler=lambda text, context=None: called() or {"text": text},
        )
        server = build_mcp_server(
            "demo", registry, argument_limits=ArgumentLimits(max_string_length=10)
        )

        result = _call(server, "echo", {"text": "x" * 11})

        assert result.isError is True
        assert result.content[0].text.startswith("Invalid arguments for tool 'echo'")
        assert "limit is 10" in result.content[0].text
        called.assert_not_called()

    def test_oversized_list_is_rejected(self):
        from pydantic import BaseModel

        from mcp_dynamic_tool_registration.server_factory import ArgumentLimits

        class TagsInput(BaseModel):
            tags: list[str] = []

        called = MagicMock()
        registry = ToolRegistry("demo")
        registry.add_tool(
            name="t",
            description="",
            input_schema=TagsInput,
            handler=lambda tags, context=None: called() or {"ok": True},
        )
        server = build_mcp_server(
            "demo", registry, argument_limits=ArgumentLimits(max_list_length=5)
        )

        result = _call(server, "t", {"tags": ["x"] * 6})

        assert result.isError is True
        assert "list-size limit" in result.content[0].text
        called.assert_not_called()

    def test_oversized_total_payload_is_rejected(self):
        from pydantic import BaseModel

        from mcp_dynamic_tool_registration.server_factory import ArgumentLimits

        class PartsInput(BaseModel):
            parts: list[str] = []

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="t",
            description="",
            input_schema=PartsInput,
            handler=lambda parts, context=None: {"ok": True},
        )
        server = build_mcp_server(
            "demo", registry, argument_limits=ArgumentLimits(max_total_chars=250)
        )

        result = _call(server, "t", {"parts": ["x" * 100, "y" * 100, "z" * 100]})

        assert result.isError is True
        assert "total argument-size limit" in result.content[0].text

    def test_within_bounds_arguments_still_dispatch(self):
        server = self._echo_server(max_string_length=100)
        result = _call(server, "echo", {"text": "hi"})

        assert result.isError is False
        assert result.structuredContent == {"text": "hi"}

    def test_custom_argument_limits_are_honored(self):
        server = self._echo_server(max_string_length=10)
        assert _call(server, "echo", {"text": "x" * 11}).isError is True
        assert _call(server, "echo", {"text": "x" * 10}).isError is False

    def test_audit_hook_is_not_invoked_for_out_of_bounds_arguments(self):
        from mcp_dynamic_tool_registration.server_factory import ArgumentLimits

        audit_calls: list[dict[str, Any]] = []

        async def audit_hook(**kwargs: Any) -> Any:
            audit_calls.append(kwargs)
            return {"audited": True}

        from pydantic import BaseModel

        class EchoInput(BaseModel):
            text: str

        registry = ToolRegistry("demo")
        registry.add_tool(
            name="echo",
            description="Echo.",
            input_schema=EchoInput,
            handler=lambda text, context=None: {"text": text},
        )
        server = build_mcp_server(
            "demo",
            registry,
            audit_hook=audit_hook,
            argument_limits=ArgumentLimits(max_string_length=10),
        )
        result = _call(server, "echo", {"text": "x" * 50})

        assert result.isError is True
        assert audit_calls == []


class TestDefaultErrorHandler:
    def test_plain_exception_is_not_handled(self):
        assert default_error_handler("t", ValueError("x")) is None

    def test_5xx_is_not_handled(self):
        assert default_error_handler("t", _HTTPError(500, "x")) is None

    def test_4xx_becomes_usage_error(self):
        result = default_error_handler("t", _HTTPError(422, "bad key"))
        assert result is not None
        assert result.isError is True
        assert result.content[0].text == "Tool 't' error: bad key"

    def test_elicitation_errors_become_usage_errors(self):
        from mcp_dynamic_tool_registration.elicitation import (
            DeclinedError,
            ElicitationNotSupportedError,
        )

        for exc in (DeclinedError("declined"), ElicitationNotSupportedError("no")):
            result = default_error_handler("t", exc)
            assert result is not None
            assert result.isError is True
