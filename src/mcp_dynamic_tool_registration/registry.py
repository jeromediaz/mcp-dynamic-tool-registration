"""Registry types: ``ToolSpec``, ``ToolRegistry``, ``McpServerRegistry``.

The registration-side data model. ``@register_tool`` collects ``ToolSpec``s
into a named ``ToolRegistry``; ``McpServerRegistry`` maps server
name/config values to their (lazily created) registries. The dispatch side
consumes ``list_tools()`` once registration is complete.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """A single registered tool, as collected by ``@register_tool``."""

    name: str
    description: str
    input_schema: type[Any] | None
    annotations: dict[str, bool]
    handler: Callable[..., Any]
    extra: dict[str, Any] = field(default_factory=dict)
    handler_signature: inspect.Signature = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.handler_signature = inspect.signature(self.handler)


class ToolRegistry:
    """Collects ``ToolSpec``s for a single named MCP server.

    ``@register_tool``'s ``wrapper(server, **params)`` calls ``add_tool`` on
    an instance of this class — never on a raw ``mcp`` SDK server object
    directly. The dispatch side consumes ``list_tools()`` to build the real
    protocol server once registration is complete.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._tools: dict[str, ToolSpec] = {}

    def add_tool(
        self,
        *,
        name: str,
        description: str,
        handler: Callable[..., Any],
        input_schema: type[Any] | None = None,
        annotations: dict[str, bool] | None = None,
        **extra: Any,
    ) -> None:
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            annotations=dict(annotations or {}),
            handler=handler,
            extra=extra,
        )

    def get_tool(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


def resolve_server_name(server_config: Any) -> str:
    """Resolve a server config value to a server name.

    A bare string is the server name itself (the common one-server-per-app
    case, e.g. ``{"tools": "demo"}``). A mapping may carry a ``server_name``
    key for future multi-server-per-module configurations.
    """
    if isinstance(server_config, str):
        return server_config
    if isinstance(server_config, Mapping):
        name = server_config.get("server_name")
        if isinstance(name, str) and name:
            return name
    raise ValueError(f"Cannot resolve MCP server name from config: {server_config!r}")


class McpServerRegistry:
    """Resolves a server name/config to its (lazily created) ``ToolRegistry``."""

    def __init__(self) -> None:
        self._registries: dict[str, ToolRegistry] = {}

    def get_or_create(self, server_config: Any) -> ToolRegistry:
        name = resolve_server_name(server_config)
        registry = self._registries.get(name)
        if registry is None:
            registry = ToolRegistry(name)
            self._registries[name] = registry
        return registry

    def get(self, name: str) -> ToolRegistry | None:
        return self._registries.get(name)

    def names(self) -> list[str]:
        return list(self._registries.keys())
