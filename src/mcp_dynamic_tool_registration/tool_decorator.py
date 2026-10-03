"""``@register_tool`` decorator for MCP tools.

Deliberately mirrors ``fastapi_dynamic_route_registration.router_decorator``'s
``@register_route`` in shape: a decorator that tags the wrapped function and
returns a ``wrapper(server, **params)`` callable, discovered later via
``inspect.getmembers``. See ``register_tools.py`` for the discovery side.
"""

from __future__ import annotations

import functools
import inspect
import types
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel


class ToolEnabledCallback(Protocol):
    def __call__(self, *args: Any, **params: Any) -> bool: ...


def _bind_params_as_defaults(
    func: Callable[..., Any], params: dict[str, Any]
) -> Callable[..., Any]:
    """Return a new callable with *params* injected as parameter defaults.

    Same mechanism as
    ``fastapi_dynamic_route_registration.router_decorator._bind_params_as_defaults``
    — kept as a local copy so this library doesn't depend on that
    FastAPI-specific package. Only params whose
    keys appear in the function signature (or the function accepts
    ``**kwargs``) are injected, so one shared kwargs dict (e.g.
    ``app_context``) can be broadcast to every tool without raising for
    tools that don't declare every key.
    """
    sig = inspect.signature(func)
    accepts_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    injectable = {
        k: v for k, v in params.items() if k in sig.parameters or accepts_var_keyword
    }

    new_params = []
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if name in injectable:
            new_params.append(param.replace(default=injectable[name]))
        else:
            new_params.append(param)

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_bound(*args: Any, **kwargs: Any) -> Any:
            for k, v in injectable.items():
                kwargs.setdefault(k, v)
            return await func(*args, **kwargs)

        async_bound.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]
        return async_bound
    else:

        @functools.wraps(func)
        def sync_bound(*args: Any, **kwargs: Any) -> Any:
            for k, v in injectable.items():
                kwargs.setdefault(k, v)
            return func(*args, **kwargs)

        sync_bound.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]
        return sync_bound


def register_tool(
    name: str | None = None,
    *,
    description: str | None = None,
    input_schema: type[BaseModel] | None = None,
    read_only_hint: bool = False,
    destructive_hint: bool = False,
    idempotent_hint: bool = False,
    open_world_hint: bool = False,
    enabled: bool | ToolEnabledCallback = True,
    **tool_kwargs: Any,
) -> Callable[..., Any]:
    """Decorator that marks a function as an MCP tool to be registered later.

    Usage mirrors ``@register_route``: decorate a handler function, and the
    decorator returns a ``wrapper(server, **params)`` callable tagged with
    ``.is_mcp_tool = True``, discovered later via ``inspect.getmembers`` (see
    ``register_tools.py``).

    Args:
        name: Tool name exposed over MCP; defaults to the function name.
        description: Tool description; defaults to the function's docstring.
        input_schema: Pydantic model used to generate the tool's JSON input
            schema.
        read_only_hint/destructive_hint/idempotent_hint/open_world_hint:
            MCP tool annotations — first-class kwargs (not buried in
            ``**tool_kwargs``) so they're discoverable at the call site.
        enabled: ``True``, ``False``, or a callable ``(**params) -> bool``
            evaluated at registration time with the per-server *params*.
        **tool_kwargs: Extra kwargs forwarded to ``server.add_tool()``.

    Returns:
        A ``Callable`` wrapper tagged with ``.is_mcp_tool = True``. When
        called as ``wrapper(server, **params)``, it registers the original
        function on *server* with *params* bound as parameter defaults via
        :func:`_bind_params_as_defaults`.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip()
        annotations = {
            "readOnlyHint": read_only_hint,
            "destructiveHint": destructive_hint,
            "idempotentHint": idempotent_hint,
            "openWorldHint": open_world_hint,
        }

        def wrapper(server: Any, **params: Any) -> None:
            if callable(enabled):
                enabled_sig = inspect.signature(enabled)
                accepts_var_keyword = any(
                    p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in enabled_sig.parameters.values()
                )
                enabled_kwargs = {
                    k: v
                    for k, v in params.items()
                    if k in enabled_sig.parameters or accepts_var_keyword
                }
                tool_enabled: bool = enabled(**enabled_kwargs)
            else:
                tool_enabled = enabled

            if not tool_enabled:
                return

            bound_func = _bind_params_as_defaults(func, params)

            server.add_tool(
                name=tool_name,
                description=tool_description,
                input_schema=input_schema,
                annotations=annotations,
                handler=bound_func,
                **tool_kwargs,
            )

        wrapper.__module__ = func.__module__
        wrapper.is_mcp_tool = True  # type: ignore[attr-defined]
        wrapper._tool_meta = {  # type: ignore[attr-defined]
            "name": tool_name,
            "description": tool_description,
            "input_schema": input_schema,
            "annotations": annotations,
        }
        return wrapper

    return decorator


def is_register_tool(func: Any) -> bool:
    return isinstance(func, types.FunctionType) and getattr(func, "is_mcp_tool", False)
