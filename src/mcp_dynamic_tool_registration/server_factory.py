"""Builds a low-level MCP ``Server`` backed by a ``ToolRegistry``.

``build_mcp_server`` wires ``list_tools``/``call_tool`` handlers to a
registry populated by ``@register_tool``, instead of the SDK's
signature-introspecting ``FastMCP.add_tool`` (which cannot take an explicit
``input_schema``). Host-specific behavior plugs in through hooks rather than
imports:

- ``current_request_context``: per-request object injected into handlers as
  the ``context`` keyword (set by the host / the ASGI app for each request).
- ``audit_hook``: wraps every tool call (e.g. to persist an audit log).
- ``error_handler``: turns expected usage errors into ``isError`` results.
- ``coerce_args``: tolerate double-encoded JSON arguments from LLM clients.

Also holds the pure helpers shared with the ASGI app: JSON argument
coercion, argument validation, the usage-error result builder, the
class-wrapped ASGI app, and the ASGI header/response helpers.

``build_streamable_http_asgi_app`` wraps the server in a bearer-token-gated
Streamable HTTP ASGI app: it validates the token through a host-supplied
``token_validator``, turns the validated payload into a per-request context
via ``context_factory`` (published on ``current_request_context`` for the
duration of ``handle_request``), and notifies ``request_hook`` for each
authenticated request. The host is responsible for entering the returned
``StreamableHTTPSessionManager``'s ``run()`` async context once, for the
lifetime of the app, before any request is handled, and for registering the
returned app with Starlette ``Route`` at both ``/x`` and ``/x/`` — *not*
``Mount``, because ``Mount`` does not forward the ASGI ``lifespan`` scope to
sub-apps, so the session manager cannot bring its own.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from typing import Any, Protocol

import pydantic
from mcp import types
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from .argument_limits import (
    DEFAULT_ARGUMENT_LIMITS,
    MAX_ARG_LIST_LENGTH,
    MAX_ARG_STRING_LENGTH,
    MAX_ARG_TOTAL_CHARS,
    ArgumentLimits,
    validate_argument_bounds,
)
from .elicitation import DeclinedError, ElicitationNotSupportedError
from .registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ARGUMENT_LIMITS",
    "MAX_ARG_LIST_LENGTH",
    "MAX_ARG_STRING_LENGTH",
    "MAX_ARG_TOTAL_CHARS",
    "ArgumentLimits",
    "AsgiApp",
    "AuditHook",
    "ContextFactory",
    "ErrorHandler",
    "PrincipalResolver",
    "RequestHook",
    "SessionGuard",
    "build_mcp_server",
    "build_streamable_http_asgi_app",
    "coerce_json_strings",
    "current_request_context",
    "default_error_handler",
    "invoke_tool",
    "usage_error_result",
    "validate_argument_bounds",
]

current_request_context: ContextVar[Any] = ContextVar(
    "mcp_dynamic_tool_registration_request_context", default=None
)
"""The current request's context object, injected into tool handlers as the
``context`` keyword. Set per request (never at registration time), so it can
carry the authenticated caller."""

type ContextFactory = Callable[[Any], Any]
"""Builds the per-request context from a validated token payload."""

type ErrorHandler = Callable[[str, Exception], types.CallToolResult | None]
"""Maps ``(tool_name, exception)`` to a tool result, or ``None`` to re-raise."""

type RequestHook = Callable[[str, str | None, str], None]
"""Called as ``(server_name, mcp_session_id, http_method)`` for each
authenticated request, before it is handled."""

type PrincipalResolver = Callable[[Any], str]
"""Returns a stable identifier of the caller (e.g. a user id) from a
validated token payload. MCP sessions are bound to it."""

type SessionGuard = Callable[[str, Any], None]
"""Called as ``(raw_bearer_token, host_context)`` on every authenticated
HTTP request, just before it is handled. Any exception it raises means
"unauthorized" (401) — unlike ``request_hook``, whose failures are logged
and never affect the request. See ``build_streamable_http_asgi_app``."""


class AuditHook(Protocol):
    """Wraps a tool call. Must perform the call itself (typically through
    :func:`invoke_tool`) and return its result."""

    def __call__(
        self,
        *,
        tool_name: str,
        spec: ToolSpec,
        context: Any,
        arguments: dict[str, Any],
        extra_handler_kwargs: dict[str, Any],
    ) -> Awaitable[Any]: ...


_BEARER_PREFIX = "bearer "


def coerce_json_strings(value: Any) -> Any:
    """Recursively coerce string values that look like JSON containers into
    their parsed form.

    Some LLM clients (e.g. opencode with certain models) double-encode
    tool-call arguments: instead of sending ``{"tags": ["a", "b"]}`` they
    send ``{"tags": "[\\"a\\", \\"b\\"]"}`` — a JSON *string* whose content
    is itself a JSON array/object.  Pydantic rejects this against a
    ``list[str]`` field with a confusing ``anyOf`` validation error.

    This walker is intentionally conservative: it only coerces a string
    when it *starts* with ``[`` or ``{`` AND parses as a valid JSON
    container (list or dict).  Plain strings, numbers, booleans, and
    strings that merely *contain* JSON-like characters are left alone.
    If the string doesn't parse as a JSON container it is returned
    unchanged so the normal Pydantic error message is still produced.
    """
    if isinstance(value, str):
        stripped = value.lstrip()
        if stripped and stripped[0] in ("[", "{"):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
            if isinstance(parsed, (list, dict)):
                return coerce_json_strings(parsed)
            return value
    elif isinstance(value, list):
        return [coerce_json_strings(item) for item in value]
    elif isinstance(value, dict):
        return {k: coerce_json_strings(v) for k, v in value.items()}
    return value


def _validate_arguments(
    input_schema: type[pydantic.BaseModel],
    arguments: dict[str, Any],
    limits: ArgumentLimits = DEFAULT_ARGUMENT_LIMITS,
) -> str | None:
    """Validate raw tool-call *arguments* against *input_schema*; return an
    error message, or ``None`` if the arguments are acceptable.

    Three checks, because tool input schemas commonly don't set
    ``model_config = ConfigDict(extra="forbid")`` and pydantic v2's default
    is ``extra="ignore"``: (1) required/mistyped fields, via
    ``model_validate`` itself, (2) unknown keys, checked separately
    against the model's field names/aliases — otherwise a typo'd kwarg
    (e.g. a misspelled field name) would validate cleanly and only
    blow up later as a raw ``TypeError`` calling the handler — and
    (3) argument *sizes* against *limits* (pydantic validates shapes,
    not magnitudes: unbounded strings/lists pass straight through), via
    :func:`validate_argument_bounds`.
    """
    try:
        input_schema.model_validate(arguments)
    except pydantic.ValidationError as exc:
        return str(exc)

    allowed = set(input_schema.model_fields)
    for f in input_schema.model_fields.values():
        if f.alias:
            allowed.add(f.alias)
    unexpected = set(arguments) - allowed
    if unexpected:
        return f"Unexpected argument(s): {', '.join(sorted(unexpected))}"

    bounds_error = validate_argument_bounds(
        arguments, limits, schema_name=input_schema.__name__
    )
    if bounds_error is not None:
        return bounds_error
    return None


def usage_error_result(tool_name: str, exc: Exception) -> types.CallToolResult:
    """Build an ``isError`` ``CallToolResult`` for an expected *usage* error
    (bad arguments, a 4xx from a tool handler, a missing elicitation
    capability, a declined confirmation) so the client sees the message and
    the exception never reaches the SDK's request handler — whose
    ``logger.exception`` a host-side error reporter's LoggingIntegration
    would otherwise capture as a production error."""
    message = getattr(exc, "detail", None) or str(exc)
    logger.info(
        "Tool %r returned a usage error (%s: %s)",
        tool_name,
        type(exc).__name__,
        message,
    )
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=f"Tool '{tool_name}' error: {message}",
            )
        ],
        isError=True,
    )


class AsgiApp:
    """Wraps a raw ASGI callable in a class instance.

    Starlette's ``Route(path, endpoint, ...)`` treats a plain function
    endpoint as ``func(request) -> response`` (via ``request_response()``),
    not raw ASGI — only a *class instance* (something that's neither
    ``inspect.isfunction`` nor ``inspect.ismethod``) is passed straight
    through as ``self.app = endpoint`` and invoked as
    ``await self.app(scope, receive, send)``. This trivial wrapper is what
    lets a host app register a Streamable HTTP handler via ``Route``
    instead of ``Mount`` — see the module docstring (and the README's "Why
    ``Route``, not ``Mount``") for why ``Mount`` doesn't work here.
    """

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self._app = app
        # Some middleware (e.g. slowapi's rate limiter) introspects the
        # matched route's endpoint via handler.__name__/__module__ to build
        # a route identifier — a bare class instance doesn't have __name__
        # (only functions/classes do), which crashes that introspection with
        # a 500 on every request. Copy identity from the wrapped function so
        # this instance looks enough like one.
        self.__name__ = getattr(app, "__name__", type(self).__name__)
        self.__module__ = getattr(app, "__module__", type(self).__module__)

    async def __call__(self, scope: Mapping[str, Any], receive: Any, send: Any) -> None:
        await self._app(scope, receive, send)


def _extract_bearer_token(scope: Mapping[str, Any]) -> str | None:
    for raw_key, raw_value in scope.get("headers", []):
        if raw_key.decode("latin-1").lower() == "authorization":
            value = raw_value.decode("latin-1")
            if value.lower().startswith(_BEARER_PREFIX):
                token = value[len(_BEARER_PREFIX) :].strip()
                return token or None
    return None


def _extract_header(scope: Mapping[str, Any], name: str) -> str | None:
    """Case-insensitive single-value header lookup on an ASGI scope.

    ``scope["headers"]`` is a list of ``(bytes, bytes)`` pairs; this mirrors
    ``_extract_bearer_token``'s scan (lowercased latin-1 decode) but returns
    the first value for an arbitrary header name. Used to read the
    ``mcp-session-id`` header for a host-side observability span tag /
    metric.
    """
    lowered = name.lower()
    for raw_key, raw_value in scope.get("headers", []):
        if raw_key.decode("latin-1").lower() == lowered:
            value = raw_value.decode("latin-1").strip()
            return value or None
    return None


async def _send_json_error(send: Any, status: int, message: str) -> None:
    body = json.dumps({"error": message}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def invoke_tool(
    spec: ToolSpec,
    context: Any,
    arguments: dict[str, Any],
    extra_handler_kwargs: dict[str, Any] | None = None,
    *,
    inject_context: bool = True,
) -> Any:
    """Default call path: ``spec.handler(context=context, **extra, **arguments)``,
    awaiting the result if it is awaitable. With ``inject_context=False`` the
    ``context`` keyword is omitted entirely, so handlers need not declare it."""
    kwargs: dict[str, Any] = {**(extra_handler_kwargs or {}), **arguments}
    if inject_context:
        kwargs["context"] = context
    result = spec.handler(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def default_error_handler(
    tool_name: str, exc: Exception
) -> types.CallToolResult | None:
    """Treat expected caller errors as ``isError`` tool results.

    - ``ElicitationNotSupportedError`` / ``DeclinedError``: a destructive tool
      whose client cannot confirm, or whose user declined.
    - Any exception with an integer-like ``status_code`` in ``[400, 500)``
      (duck-typed, so e.g. a web framework's ``HTTPException`` works without
      this library importing that framework).

    Everything else returns ``None``, meaning "re-raise": the SDK then turns
    it into its own error result. Converting usage errors here keeps them out
    of the SDK's exception logging, where error trackers would report them as
    server faults."""
    if isinstance(exc, ElicitationNotSupportedError | DeclinedError):
        return usage_error_result(tool_name, exc)
    status_code = getattr(exc, "status_code", None)
    if status_code is not None and 400 <= int(status_code) < 500:
        return usage_error_result(tool_name, exc)
    return None


def _is_context_manager(value: Any) -> bool:
    return hasattr(type(value), "__enter__") and hasattr(type(value), "__exit__")


def build_mcp_server(
    name: str,
    tool_registry: ToolRegistry,
    *,
    audit_hook: AuditHook | None = None,
    error_handler: ErrorHandler | None = default_error_handler,
    coerce_args: bool = True,
    inject_context: bool = True,
    argument_limits: ArgumentLimits | None = None,
) -> Server:
    """Build a low-level MCP ``Server`` whose tool list/dispatch are backed
    by *tool_registry*.

    Args:
        name: Server name reported to clients.
        tool_registry: Source of the tools (read on every request, so tools
            registered later are served too).
        audit_hook: Optional wrapper around every tool call; when set it is
            responsible for calling the handler (see :class:`AuditHook`).
        error_handler: Maps handler exceptions to tool results; ``None``
            disables the mapping (every exception propagates to the SDK).
        coerce_args: Parse JSON-container strings in arguments before
            validation (see :func:`coerce_json_strings`).
        inject_context: Pass ``current_request_context`` to handlers as the
            ``context`` keyword. Disable for handlers that take no context.
        argument_limits: Size bounds enforced on tool-call arguments after
            coercion and schema validation (see
            :func:`validate_argument_bounds`); ``None`` means
            :data:`DEFAULT_ARGUMENT_LIMITS`. Bounds are enforced
            server-side only — they are deliberately *not* advertised in
            the tools' ``inputSchema`` (``model_json_schema``), so a host
            can tighten them without changing what clients see.
    """
    limits = argument_limits or DEFAULT_ARGUMENT_LIMITS
    server: Server = Server(name)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=(
                    spec.input_schema.model_json_schema()
                    if spec.input_schema is not None
                    else {"type": "object", "properties": {}}
                ),
                annotations=(
                    types.ToolAnnotations(
                        readOnlyHint=spec.annotations.get("readOnlyHint", False),
                        destructiveHint=spec.annotations.get("destructiveHint", False),
                        idempotentHint=spec.annotations.get("idempotentHint", False),
                        openWorldHint=spec.annotations.get("openWorldHint", False),
                    )
                    if spec.annotations
                    else None
                ),
            )
            for spec in tool_registry.list_tools()
        ]

    # validate_input=False: arguments are validated below with Pydantic, after
    # JSON-string coercion. The SDK's jsonschema validation would reject
    # double-encoded arguments before they could be coerced.
    @server.call_tool(validate_input=False)
    async def _call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
        spec = tool_registry.get_tool(tool_name)
        if spec is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Coerce before both validation and the handler call so the coerced
        # values are what actually reach the handler.
        if coerce_args:
            arguments = coerce_json_strings(arguments)

        # Malformed arguments (unknown/missing/mistyped fields) are a caller
        # mistake, not a server fault: answer with a normal isError result
        # instead of letting a TypeError from the handler call reach the SDK.
        if spec.input_schema is not None:
            validation_error = _validate_arguments(spec.input_schema, arguments, limits)
            if validation_error is not None:
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"Invalid arguments for tool '{tool_name}': "
                            f"{validation_error}",
                        )
                    ],
                    isError=True,
                )

        # Per-request context (e.g. the authenticated caller), never a
        # registration-time default. When it is a context manager it is
        # entered for the handler's duration, which lets hosts activate it
        # (e.g. through their own contextvar) across the awaits below.
        context: Any = current_request_context.get() if inject_context else None

        # Handlers that declare an ``mcp_session`` parameter (e.g. to call
        # confirm_destructive) receive the live ServerSession; it is never
        # passed to handlers that do not declare it.
        extra_handler_kwargs: dict[str, Any] = {}
        if "mcp_session" in spec.handler_signature.parameters:
            extra_handler_kwargs["mcp_session"] = server.request_context.session

        async def _invoke() -> Any:
            if audit_hook is not None:
                return await audit_hook(
                    tool_name=tool_name,
                    spec=spec,
                    context=context,
                    arguments=arguments,
                    extra_handler_kwargs=extra_handler_kwargs,
                )
            return await invoke_tool(
                spec,
                context,
                arguments,
                extra_handler_kwargs,
                inject_context=inject_context,
            )

        try:
            if _is_context_manager(context):
                with context:
                    return await _invoke()
            return await _invoke()
        except Exception as exc:
            if error_handler is not None:
                result = error_handler(tool_name, exc)
                if result is not None:
                    return result
            raise

    return server


def build_streamable_http_asgi_app(
    name: str,
    tool_registry: ToolRegistry,
    *,
    token_validator: Callable[[str], Any],
    context_factory: ContextFactory | None = None,
    audit_hook: AuditHook | None = None,
    error_handler: ErrorHandler | None = default_error_handler,
    coerce_args: bool = True,
    request_hook: RequestHook | None = None,
    principal_of: PrincipalResolver | None = None,
    session_guard: SessionGuard | None = None,
    argument_limits: ArgumentLimits | None = None,
) -> tuple[AsgiApp, StreamableHTTPSessionManager]:
    """Build a bearer-token-gated Streamable HTTP ASGI app for one MCP server.

    Returns ``(asgi_app, session_manager)`` — the caller is responsible for
    entering *session_manager*'s ``run()`` context for the host app's
    lifetime and registering *asgi_app* as a route (see the module docstring).

    Args:
        name: Server name reported to clients and to *request_hook*.
        tool_registry: Source of the tools served by the app.
        token_validator: Maps the bearer token to a payload object; any
            exception it raises means "unauthorized" (401).
        context_factory: Builds the per-request context from the validated
            payload; ``None`` means no context is published and handlers are
            built without context injection.
        audit_hook: Passed through to :func:`build_mcp_server`.
        error_handler: Passed through to :func:`build_mcp_server`.
        coerce_args: Passed through to :func:`build_mcp_server`.
        request_hook: Called as ``(server_name, mcp_session_id,
            http_method)`` for each authenticated request, just before it is
            handled; failures are logged and never affect the request.
        principal_of: Maps the validated payload to a stable caller id
            (e.g. a user id). Each MCP session is bound to the caller that
            created it: a request carrying another caller's credentials and
            an existing ``mcp-session-id`` is answered as if the session did
            not exist (404). Defaults to a SHA-256 of the bearer token, which
            binds a session to the exact token that created it. An exception
            raised here is treated like an invalid token (401).
        session_guard: Called as ``(raw_bearer_token, host_context)`` on
            EVERY HTTP request that reaches the ASGI app after the token
            validates — including follow-up requests on an already-open
            session, whose JSON-RPC frames are otherwise forwarded to the
            session's persistent task without any re-validation. Unlike
            *request_hook*, an exception raised here rejects the request:
            it is answered like an invalid token (401). Use it to re-check
            per-request authorization state (token revocation, caller
            status) for the lifetime of a session, not just at
            ``initialize``. ``None`` (the default) guards nothing.
        argument_limits: Passed through to :func:`build_mcp_server`.
    """
    mcp_server = build_mcp_server(
        name,
        tool_registry,
        audit_hook=audit_hook,
        error_handler=error_handler,
        coerce_args=coerce_args,
        inject_context=context_factory is not None,
        argument_limits=argument_limits,
    )
    # Stateful (the SDK default): a session's ServerSession persists across
    # requests, so client_params negotiated at `initialize` survive into
    # later tool calls — required for check_client_capability() (elicitation)
    # to see anything but a blank slate. Stateless mode creates a brand new
    # ServerSession per HTTP request with client_params=None, which makes
    # every capability check fail unconditionally.
    #
    # This is compatible with the per-request `current_request_context`
    # plumbing below: a new session's persistent app.run() task is spawned
    # (via the SDK's task group) from within asgi_app's call stack, at a
    # point after current_request_context.set() has already run for that
    # (session-creating) request. anyio task spawning copies the spawning
    # coroutine's contextvars at that instant, so the persistent task keeps
    # that snapshot for its whole lifetime. Follow-up requests to the same
    # session don't re-run the handler in the per-request task at all (they
    # just forward bytes to the already-running session task), so they don't
    # need to re-set the contextvar. That is only safe because a session is
    # bound to one caller for its lifetime: the SDK rejects a request whose
    # ``scope["user"]`` principal differs from the session creator's, which
    # is why asgi_app publishes the caller as an SDK ``AuthenticatedUser``.
    session_manager = StreamableHTTPSessionManager(app=mcp_server, stateless=False)

    async def asgi_app(scope: Mapping[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return

        token = _extract_bearer_token(scope)
        if token is None:
            await _send_json_error(send, 401, "Missing bearer token")
            return

        try:
            token_payload = token_validator(token)
            principal = (
                principal_of(token_payload)
                if principal_of is not None
                else hashlib.sha256(token.encode("utf-8")).hexdigest()
            )
        except Exception:
            logger.info(
                "MCP request rejected: token validation failed for server=%s", name
            )
            await _send_json_error(send, 401, "Invalid or unauthorized token")
            return

        # The SDK binds each session to the principal found in scope["user"]
        # (see StreamableHTTPSessionManager) and rejects other callers.
        scope = {
            **scope,
            "user": AuthenticatedUser(
                AccessToken(
                    token=token, client_id=principal, scopes=[], subject=principal
                )
            ),
        }

        var_token = (
            current_request_context.set(context_factory(token_payload))
            if context_factory is not None
            else None
        )
        # When a session guard is configured, the per-request context is
        # published BEFORE it runs (and reset on every exit path) so the
        # guard can inspect the live MCP frame — e.g. reject a tool call
        # whose caller was deactivated since the session opened. The
        # session's own persistent task keeps the snapshot it inherited
        # when the session was created; this per-request set/reset never
        # mutates it. Without a guard, the behavior is unchanged: the
        # context is published right before the request is handled.
        try:
            if session_guard is not None:
                session_guard(token, current_request_context.get())
        except Exception:
            if var_token is not None:
                current_request_context.reset(var_token)
            logger.info(
                "MCP request rejected: session guard declined for server=%s", name
            )
            await _send_json_error(send, 401, "Invalid or unauthorized token")
            return

        try:
            if request_hook is not None:
                try:
                    request_hook(
                        name,
                        _extract_header(scope, "mcp-session-id"),
                        scope.get("method", ""),
                    )
                except Exception:
                    logger.exception("request_hook failed for server=%s", name)
            await session_manager.handle_request(scope, receive, send)
        finally:
            if var_token is not None:
                current_request_context.reset(var_token)

    return AsgiApp(asgi_app), session_manager
