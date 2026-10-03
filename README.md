# MCP Dynamic Tool Registration

[![PyPI - License](https://img.shields.io/pypi/l/mcp-dynamic-tool-registration)](https://pypi.org/project/mcp-dynamic-tool-registration/)
[![PyPI - Version](https://img.shields.io/pypi/v/mcp-dynamic-tool-registration)](https://pypi.org/project/mcp-dynamic-tool-registration/)

Decorator-based, registry-first dynamic tool registration for MCP (Model Context
Protocol) servers. Declare tool handlers with a decorator, collect them into a
registry from a configuration map, and serve them over a bearer-token
authenticated Streamable HTTP ASGI app — without importing any host application
framework.

It is the MCP-side sibling of
[fastapi-dynamic-route-registration](https://github.com/jeromediaz/fastapi-dynamic-route-registration):
handler modules decorate their functions with `@register_tool`, and the host
application mounts whole modules from a configuration map — with per-server
params injected as parameter defaults. The registration half is deliberately a
registry, not a live protocol server: the library builds the MCP server from the
registry only when it serves (or you call handlers in-process), and every
host-specific concern — auditing, error mapping, observability, per-request
context — plugs in through a hook instead of an import.

```bash
pip install mcp-dynamic-tool-registration
```

## How to Use

### A module of tools

`myapp/mcp/tools.py` (a plain module in a plain package — nothing to import
from the library except the decorator):

```python
from mcp_dynamic_tool_registration import register_tool


@register_tool("echo", description="Echo a message back.", read_only_hint=True)
def echo_tool(message: str, context=None):
    return {"echo": message}


@register_tool("ping")
def ping_tool():
    """Ping."""
    return {"pong": True}
```

`@register_tool` accepts:

- `name` and `description` (the description defaults to the docstring);
- `input_schema`, a Pydantic model used to generate the tool's JSON schema
  (call arguments are validated against it before the handler runs);
- the MCP tool annotations `read_only_hint`, `destructive_hint`,
  `idempotent_hint`, `open_world_hint`;
- `enabled`: `True`, `False`, or a callable `(**params) -> bool` evaluated at
  registration time with the per-server params;
- `**tool_kwargs`: anything else, passed through to the registry's
  `add_tool()` and stored in `ToolSpec.extra`. The library never reads
  `extra` itself — it is an opaque side-channel for host hooks (for example an
  `audit_model` key that an `audit_hook` dispatches on).

The handler contract: an optional `context` keyword — the library injects the
current request's context object there (see *Why `context` is per-request, not
a registration-time default* below) — and an optional `mcp_session` keyword,
which is injected **only when the handler declares it**.

### Registering a module

`register_tools` takes a server registry, a common dotted module prefix, and a
`{module_suffix: server_config}` map — the same shape the FastAPI sibling uses
for `{module_suffix: prefix}`, except the value names an MCP server. In a real
app `myapp` and `myapp.mcp` are ordinary packages on your path and
`myapp/mcp/tools.py` is the module from the previous snippet; the `tools`
module below stands in for that file so the snippet runs as is:

```python
import sys
import types

from mcp_dynamic_tool_registration import McpServerRegistry, register_tool, register_tools


@register_tool("echo", description="Echo a message back.", read_only_hint=True)
def echo_tool(message: str, context=None):
    return {"echo": message}


@register_tool("ping")
def ping_tool():
    """Ping."""
    return {"pong": True}


# The stand-in for the package layout (in a real app myapp/ and myapp/mcp/
# are ordinary packages on your path and the module above is
# myapp/mcp/tools.py). Aliasing this script's own module is what makes the
# discovery filter — fn.__module__ == module.__name__ — find the tools.
pkg, mcp_pkg = types.ModuleType("myapp"), types.ModuleType("myapp.mcp")
mcp_pkg.__path__ = []
sys.modules.update({"myapp": pkg, "myapp.mcp": mcp_pkg, "myapp.mcp.tools": sys.modules[__name__]})

registry = McpServerRegistry()
register_tools(registry, "myapp.mcp", {"tools": "demo"}, {})

tool_registry = registry.get("demo")  # ToolRegistry for the "demo" server
print([spec.name for spec in tool_registry.list_tools()])
# ['echo', 'ping']
```

A bare-string config is the server name; a mapping may carry a `server_name`
key (see `resolve_server_name`). A module that fails to import or register is
logged at ERROR and skipped — the remaining modules still register. The fourth
argument (`tool_kwargs`) binds shared values as default parameter values on
every discovered tool; pass `{}` unless you need it.

The decoration side never touches a protocol server: the decorator tags the
function and returns a `wrapper(server, **params)` callable, which
`register_tool_module` discovers via `inspect.getmembers` and calls. You can
call it yourself with any object exposing `add_tool(...)`. `is_register_tool`
is the discovery predicate.

### In-process use

The registry is usable without any HTTP server — handy for agents that resolve
tools in-process (continuing from the previous snippet's `tool_registry`):

```python
spec = tool_registry.get_tool("echo")
print(spec.handler(context=None, message="hello"))
# {'echo': 'hello'}
```

In a script you would write exactly that once, after `register_tools`; the
registry, not the protocol server, is the always-available surface.

### Serving over Streamable HTTP

`build_streamable_http_asgi_app` returns `(asgi_app, session_manager)`. The
`token_validator` maps a bearer token to a payload object — any exception it
raises means 401 — and the `context_factory` turns that payload into the
per-request context published for the duration of the request.

```python
import sys
import types
from contextlib import asynccontextmanager

from mcp_dynamic_tool_registration import (
    McpServerRegistry,
    build_streamable_http_asgi_app,
    register_tool,
    register_tools,
)
from starlette.routing import Route


# The same tools as myapp/mcp/tools.py. In a real app that module is an
# ordinary package file; here the script aliases itself under that name —
# the module filter (fn.__module__ == module.__name__) matches the
# decorator-stamped __module__ of the tools defined right above.
@register_tool("echo", description="Echo a message back.", read_only_hint=True)
def echo_tool(message: str, context=None):
    return {"echo": message}


@register_tool("ping")
def ping_tool():
    """Ping."""
    return {"pong": True}


pkg, mcp_pkg = types.ModuleType("myapp"), types.ModuleType("myapp.mcp")
mcp_pkg.__path__ = []
sys.modules.update({"myapp": pkg, "myapp.mcp": mcp_pkg, "myapp.mcp.tools": sys.modules[__name__]})

registry = McpServerRegistry()
register_tools(registry, "myapp.mcp", {"tools": "demo"}, {})
print([spec.name for spec in registry.get("demo").list_tools()])
# ['echo', 'ping']

def validate_token(token: str) -> dict:
    # Raise to reject: any exception means 401. A return value (even None)
    # means the token is valid and becomes the payload.
    if token != "s3cret":
        raise PermissionError("invalid token")
    return {"uid": "u-1"}


asgi_app, session_manager = build_streamable_http_asgi_app(
    "demo",
    registry.get_or_create("demo"),
    token_validator=validate_token,
    context_factory=lambda payload: payload,
)


# The host app must enter the session manager's run() context once, for the
# app's whole lifetime (shown here with a Starlette lifespan):
@asynccontextmanager
async def lifespan(app):
    async with session_manager.run():
        yield


# Route, never Mount, at both the bare path and the trailing-slash path —
# see "Why Route, not Mount" below.
routes = [
    Route("/mcp/demo", endpoint=asgi_app, methods=["GET", "POST", "DELETE"]),
    Route("/mcp/demo/", endpoint=asgi_app, methods=["GET", "POST", "DELETE"]),
]
```

Requests without a valid bearer token get a JSON `401` before the session
manager is ever reached. `build_mcp_server(name, tool_registry)` reads the
registry on every request, so tools registered later are served too.

### Confirming destructive calls

`confirm_destructive` asks the connected MCP client to confirm a destructive
action via elicitation. It **fails closed**: a client that did not negotiate
elicitation support at `initialize` raises `ElicitationNotSupportedError`; a
declined or cancelled prompt, or a client that does not answer within 60
seconds, raises `DeclinedError`. Only an explicit accept returns `True`.

```python
import asyncio

from mcp_dynamic_tool_registration import (
    ElicitationNotSupportedError,
    confirm_destructive,
    register_tool,
)


@register_tool("delete_thing", destructive_hint=True)
async def delete_thing(thing_id: str, mcp_session):
    """Delete a thing, but only after the client confirms."""
    confirmed = await confirm_destructive(mcp_session, "Really delete it?")
    return {"deleted": thing_id if confirmed else None}


# Declaring mcp_session is all it takes: the library injects the live
# ServerSession for the current call, and nothing at all reaches handlers
# that do not declare it. The session is scoped to one authenticated
# identity for its whole lifetime, which is what makes confirmations safe
# to attach to it.
#
# Fails closed, so a host can tell "the client cannot confirm" from a real
# handler failure:
async def demo():
    class NoElicitationSession:
        def check_client_capability(self, capability):
            return False

    try:
        await confirm_destructive(NoElicitationSession(), "Really delete it?")
    except ElicitationNotSupportedError:
        print("client cannot confirm -> refused")


asyncio.run(demo())
# client cannot confirm -> refused
```

### The hooks reference

Every host-specific behavior is a keyword argument of the builders — the
library imports nothing from your framework.

| Hook | Signature | Purpose |
|------|-----------|---------|
| `audit_hook` | `async (*, tool_name, spec, context, arguments, extra_handler_kwargs) -> result` | Wraps **every** tool call. When set, it is responsible for performing the call itself, typically through `invoke_tool(spec, context, arguments, extra_handler_kwargs)`. Read `spec.extra` here to dispatch per-tool behavior. |
| `error_handler` | `(tool_name, exc) -> CallToolResult \| None` | Maps handler exceptions to `isError` results; return `None` to re-raise to the SDK. Defaults to `default_error_handler`: elicitation errors and any exception with a duck-typed `status_code` in `[400, 500)` become `usage_error_result(...)` (text prefixed `Tool '<name>' error: `), everything else propagates. `None` disables the mapping entirely. |
| `coerce_args` | `bool` (default `True`) | Runs `coerce_json_strings` on the call arguments before validation: some LLM clients double-encode containers (`"[\"a\", \"b\"]"` instead of `["a", "b"]`); strings that *start* with `[` or `{` **and** parse as JSON are decoded, anything else is left alone so the normal validation error still happens. |
| `request_hook` | `(server_name, mcp_session_id, http_method) -> None` | ASGI app only: called for each **authenticated** request, after the token validates and after the context is set, just before `handle_request`. Detection only (metrics/spans); exceptions are logged and swallowed so a broken hook never affects the request. |
| `context_factory` | `(token_payload) -> context` | ASGI app only: builds the per-request context from the validated payload and publishes it on `current_request_context`. `None` means no context is published and handlers are built without `context` injection (`inject_context=False` on `build_mcp_server`). |

The types `AuditHook`, `ContextFactory`, `ErrorHandler` and `RequestHook` are
exported for annotating your own wrappers. `current_request_context` is the
underlying `ContextVar`, exposed for hosts (and tests) that want to inspect or
set it directly.

The public API is exactly these names:

```python
import mcp_dynamic_tool_registration as m

print(sorted(m.__all__))
# ['AsgiApp', 'AuditHook', 'ContextFactory', 'DeclinedError',
#  'ElicitationNotSupportedError', 'ErrorHandler', 'McpServerRegistry',
#  'RequestHook', 'ServerRegistry', 'ToolEnabledCallback', 'ToolRegistry',
#  'ToolSpec', 'build_mcp_server', 'build_streamable_http_asgi_app',
#  'coerce_json_strings', 'confirm_destructive', 'current_request_context',
#  'default_error_handler', 'invoke_tool', 'is_register_tool', 'register_tool',
#  'register_tool_module', 'register_tools', 'resolve_server_name',
#  'usage_error_result']
```

A complete hooks example — an `audit_hook` that dispatches on the
`audit_model` each tool carries in `**tool_kwargs`, called the way the SDK's
request handler calls it:

```python
import asyncio
import types as pytypes

from mcp import types
from mcp_dynamic_tool_registration import (
    ToolRegistry,
    build_mcp_server,
    invoke_tool,
)


# The bound handler register_tools produces for myapp/mcp/tools.py's echo
# tool, written out directly so this script runs as is (its docstring is the
# description @register_tool would have captured).
async def echo_handler(message: str, context=None):
    """Echo a message back."""
    return {"echo": message}


def audit_hook_for(audit_log):
    async def audit_hook(*, tool_name, spec, context, arguments, extra_handler_kwargs):
        audit_model = spec.extra.get("audit_model")  # what the tool declared
        audit_log.append((tool_name, audit_model))
        return await invoke_tool(spec, context, arguments, extra_handler_kwargs)

    return audit_hook


async def main():
    # (In a real app: register_tools(McpServerRegistry(), "myapp.mcp",
    # {"tools": "demo"}, {}) — here the tool lands in the registry directly.)
    tool_registry = ToolRegistry("demo")
    tool_registry.add_tool(
        name="echo",
        description=echo_handler.__doc__.strip(),
        handler=echo_handler,
        audit_model={"tool": "echo"},  # lands in ToolSpec.extra, opaque here
    )

    audit_log = []
    server = build_mcp_server(
        "demo",
        tool_registry,
        audit_hook=audit_hook_for(audit_log),
        # error_handler=default_error_handler (the default) maps elicitation
        # errors and duck-typed 4xx exceptions to isError results;
        # coerce_args=True (the default) repairs double-encoded arguments;
        # inject_context=False drops the context kwarg entirely.
    )

    handler = server.request_handlers[types.CallToolRequest]
    result = await handler(
        types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="echo", arguments={"message": "hello"}
            ),
        )
    )
    print(result.root.structuredContent, "| audit log:", audit_log)


asyncio.run(main())
# {'echo': 'hello'} | audit log: [('echo', {'tool': 'echo'})]
```

### Why `context` is per-request, not a registration-time default

`register_tools(..., tool_kwargs)` binds shared values as parameter defaults
once, at startup — fine for application-wide objects, wrong for anything that
depends on who is calling. The caller's identity is only known per request, so
it travels differently: the ASGI app builds it with `context_factory` from the
validated token, publishes it on `current_request_context`, and the dispatcher
passes it to the handler as the `context` keyword for that call only. A
handler therefore always sees the current request's caller, never a stale
value captured at registration time — and in tests or in-process use you
simply pass `context=...` yourself.

The same reasoning applies to `mcp_session`: it is injected per call, only for
handlers that declare it, which also makes it trivial to fake in unit tests.

### Why `Route`, not `Mount`

Register the ASGI app with Starlette `Route` at both `/mcp/demo` and
`/mcp/demo/` — *not* `Mount` — for two reasons:

1. `Mount` does not forward the ASGI `lifespan` scope to sub-apps, so the
   session manager cannot enter its own `run()` context; the host must enter it
   at the app level and stash the manager accordingly.
2. `Mount`'s path matching structurally requires at least the trailing slash
   (`Mount.path_regex` is compiled as `self.path + "/{path:path}"`, which never
   matches the bare mount path). A request to the bare path gets a 307
   redirect-with-slash from Starlette's router instead of a direct match, and
   real MCP clients (confirmed with Claude Desktop) send the exact URL
   configured in the connector without following that redirect for a streaming
   POST, retrying forever. Two explicit `Route`s for the same handler
   sidesteps the whole redirect path.

Passing a `Route` a raw ASGI app works because Starlette passes a *class
instance* endpoint straight through as `self.app`; that is exactly what the
exported `AsgiApp` wrapper (which also copies `__name__`/`__module__` from the
wrapped callable so introspection-based middleware doesn't crash) is for.

### How this compares to FastMCP / fastapi-mcp

FastMCP registers tools onto a live server object as they are defined and
derives their schemas from function signatures; fastapi-mcp goes the other way
and exposes existing FastAPI endpoints as tools. This library takes a third
approach:

- **Registry first.** `@register_tool` never touches a protocol server; tools
  land in a plain `ToolRegistry` you can inspect, assert on, or call
  in-process. The MCP `Server` is built from the registry at serve time and
  re-reads it on every request.
- **Explicit input schemas.** A handler's schema comes from a Pydantic model
  you pass (`input_schema=`), not from signature magic — so a handler can take
  injected plumbing (`context`, `mcp_session`) that clients never see.
- **Configuration-driven mounting.** Which modules belong to which server is a
  data map (`{module_suffix: server_config}`), easy to keep in a JSON file per
  host application — the same pattern as the FastAPI sibling.
- **Hooks, not forks.** Auditing, error policy, argument coercion and request
  observability are constructor arguments. A host with its own error tracker or
  audit store configures the library instead of patching it.
- **Plain ASGI, no web framework.** The HTTP half depends on nothing but the
  `mcp` SDK: it mounts under Starlette, FastAPI, or any ASGI server.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/jeromediaz/mcp-dynamic-tool-registration/blob/main/LICENSE) file for details.
