"""Discovery/registration of ``@register_tool``-decorated functions.

Mirrors ``fastapi_dynamic_route_registration.register_routers``: each key in
a configuration map is a module suffix (appended to a common
``module_prefix``) whose value names the MCP server the module's tools should
be registered onto.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from inspect import getmembers
from typing import Any, Protocol

from .tool_decorator import is_register_tool

logger = logging.getLogger(__name__)


class ServerRegistry(Protocol):
    """Structural protocol for whatever resolves a server name to a server
    object capable of ``add_tool(...)`` — implemented by
    ``registry.McpServerRegistry``.
    """

    def get_or_create(self, server_config: Any) -> Any: ...


def register_tool_module(
    server: Any,
    module_name: str,
    tool_kwargs: Mapping[str, Any],
) -> list[str]:
    """Register every ``@register_tool``-decorated function found in
    *module_name* onto *server*.

    Args:
        server: The MCP server (or server-like object) to register tools on
            — passed through to each tool's ``wrapper(server, **params)``.
        module_name: Dotted import path of the module to scan.
        tool_kwargs: Extra key/value pairs bound as default parameter values
            in every discovered tool function (e.g. a shared application context).

    Returns:
        The names of the functions that were registered (empty list if the
        module contains no ``@register_tool``-decorated functions).
    """
    module_object = __import__(module_name, fromlist=[""])
    # Only register functions actually defined in this module — a tool
    # re-exported into the module's namespace (e.g. ``from module_b import
    # some_tool``) must not be registered here, or it would be registered
    # twice when module_b's own pass also finds it. Compare against the
    # module's canonical ``__name__`` (not the import path used to look it
    # up), so the same module registered under an alias still matches its
    # own functions.
    functions = [
        (name, fn)
        for name, fn in getmembers(module_object, is_register_tool)
        if fn.__module__ == module_object.__name__
    ]

    if not functions:
        return []

    for _name, register_fn in functions:
        register_fn(server, **tool_kwargs)

    return [name for name, _fn in functions]


def register_tools(
    server_registry: ServerRegistry,
    module_prefix: str,
    servers: Mapping[str, Any],
    tool_kwargs: Mapping[str, Any],
) -> None:
    """Register every module named in *servers* onto its declared server.

    Args:
        server_registry: Resolves a server-name/config value to the server
            object tools should be registered on.
        module_prefix: Common dotted prefix for all modules (e.g.
            ``"myapp.mcp"``).
        servers: Mapping of ``{module_suffix: server_config}`` — a
            ``{module_suffix: prefix}``-style map where the value
            names/configures an MCP server rather than a URL prefix.
        tool_kwargs: Extra key/value pairs bound as default parameter values
            in every discovered tool (e.g. a shared context object).

    A failure to register one module (import error, malformed tool
    decorator, ...) is logged at ERROR level and skipped so that the
    remaining modules are still registered.
    """
    for module_suffix, server_config in servers.items():
        module_name = f"{module_prefix}.{module_suffix}"
        try:
            server = server_registry.get_or_create(server_config)
            register_tool_module(server, module_name, tool_kwargs)
        except Exception:
            logger.exception(
                "Failed to register MCP tool module %r (server config: %r); "
                "continuing with remaining modules.",
                module_name,
                server_config,
            )
