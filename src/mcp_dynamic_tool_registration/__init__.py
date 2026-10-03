"""MCP Dynamic Tool Registration — registry-first tool registration for MCP servers."""

from .elicitation import (
    DeclinedError,
    ElicitationNotSupportedError,
    confirm_destructive,
)
from .register_tools import (
    ServerRegistry,
    register_tool_module,
    register_tools,
)
from .registry import (
    McpServerRegistry,
    ToolRegistry,
    ToolSpec,
    resolve_server_name,
)
from .server_factory import (
    AsgiApp,
    AuditHook,
    ContextFactory,
    ErrorHandler,
    PrincipalResolver,
    RequestHook,
    build_mcp_server,
    build_streamable_http_asgi_app,
    coerce_json_strings,
    current_request_context,
    default_error_handler,
    invoke_tool,
    usage_error_result,
)
from .tool_decorator import (
    ToolEnabledCallback,
    is_register_tool,
    register_tool,
)

__version__ = "0.1.0"

__all__: list[str] = [
    "AsgiApp",
    "AuditHook",
    "ContextFactory",
    "DeclinedError",
    "ElicitationNotSupportedError",
    "ErrorHandler",
    "McpServerRegistry",
    "PrincipalResolver",
    "RequestHook",
    "ServerRegistry",
    "ToolEnabledCallback",
    "ToolRegistry",
    "ToolSpec",
    "build_mcp_server",
    "build_streamable_http_asgi_app",
    "coerce_json_strings",
    "confirm_destructive",
    "current_request_context",
    "default_error_handler",
    "invoke_tool",
    "is_register_tool",
    "register_tool",
    "register_tool_module",
    "register_tools",
    "resolve_server_name",
    "usage_error_result",
]
