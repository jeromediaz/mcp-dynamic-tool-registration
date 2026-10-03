"""Tests for register_tool_module()/register_tools() against a synthetic module.

Mirrors fastapi_dynamic_route_registration's register_router/register_routers
test coverage: module-suffix -> server resolution, empty-module no-op.
"""

from __future__ import annotations

import sys
import types

from mcp_dynamic_tool_registration.register_tools import (
    register_tool_module,
    register_tools,
)
from mcp_dynamic_tool_registration.registry import McpServerRegistry
from mcp_dynamic_tool_registration.tool_decorator import register_tool

# ---------------------------------------------------------------------------
# Synthetic module of @register_tool-decorated functions (this module itself)
# ---------------------------------------------------------------------------


@register_tool("echo", description="Echo params.")
def echo_tool(app_context=None):
    return {"app_context": app_context}


@register_tool("ping")
def ping_tool():
    """Ping."""
    return {"pong": True}


class TestRegisterToolModule:
    def test_registers_functions_found_in_module(self):
        registry = McpServerRegistry()
        tool_registry = registry.get_or_create("demo")

        names = register_tool_module(tool_registry, __name__, {"app_context": "gc"})

        assert set(names) == {"echo_tool", "ping_tool"}
        assert tool_registry.get_tool("echo") is not None
        assert tool_registry.get_tool("ping") is not None

    def test_tool_kwargs_bound_as_defaults(self):
        registry = McpServerRegistry()
        tool_registry = registry.get_or_create("demo")

        register_tool_module(tool_registry, __name__, {"app_context": "the-ctx"})

        spec = tool_registry.get_tool("echo")
        assert spec is not None
        assert spec.handler() == {"app_context": "the-ctx"}

    def test_reexported_tool_not_registered_twice(self):
        """A tool re-exported into module A from module B must only be
        registered by module B's own pass, not by both passes.

        The tool is defined in this test module (so its wrapper's
        ``__module__`` is ``__name__`` here), then placed into two synthetic
        modules: module B (the "defining" side, via a ``__module__``
        override to simulate definition there) and module A (a mere
        re-export, no override).
        """
        module_a = types.ModuleType("_mdtr_reexport_a")
        module_b = types.ModuleType("_mdtr_reexport_b")

        @register_tool("reexported")
        def reexported_tool():
            return {"ok": True}

        # Simulate the tool being defined in module_b (the decorator sets
        # wrapper.__module__ to the wrapped function's __module__, which is
        # this test module; override it to module_b to mimic a tool whose
        # decorated function lives in module_b).
        reexported_tool.__module__ = module_b.__name__
        module_b.reexported_tool = reexported_tool
        module_a.reexported_tool = reexported_tool  # mere re-export

        sys.modules[module_a.__name__] = module_a
        sys.modules[module_b.__name__] = module_b
        try:
            registry = McpServerRegistry()
            tool_registry = registry.get_or_create("demo")

            names_a = register_tool_module(tool_registry, module_a.__name__, {})
            names_b = register_tool_module(tool_registry, module_b.__name__, {})

            assert names_a == []
            assert names_b == ["reexported_tool"]
            assert len(tool_registry) == 1
        finally:
            del sys.modules[module_a.__name__]
            del sys.modules[module_b.__name__]

    def test_empty_module_returns_empty_list(self):
        empty = types.ModuleType("_mdtr_empty_test_module")
        sys.modules["_mdtr_empty_test_module"] = empty
        try:
            registry = McpServerRegistry()
            tool_registry = registry.get_or_create("demo")
            names = register_tool_module(tool_registry, "_mdtr_empty_test_module", {})
            assert names == []
            assert len(tool_registry) == 0
        finally:
            del sys.modules["_mdtr_empty_test_module"]


class TestRegisterTools:
    """register_tools() using functions defined in this test module.

    Mirrors test_register_route.py's test_register_routers: this module is
    registered under a synthetic "pkg.mod" name in sys.modules so
    register_tools' `f"{module_prefix}.{module_suffix}"` import path
    resolves back to this already-imported module, regardless of whether
    pytest happened to import this file as a top-level or dotted module.
    """

    def test_module_suffix_maps_to_named_server(self):
        server_registry = McpServerRegistry()
        # Build module_prefix/suffix so f"{prefix}.{suffix}" == __name__.
        aliased = "." not in __name__
        if aliased:
            # Top-level module: alias it under a dotted name so the
            # "prefix.suffix" convention still resolves via sys.modules.
            prefix, suffix = "_mdtr_alias_pkg", "mod"
            sys.modules[f"{prefix}.{suffix}"] = sys.modules[__name__]
        else:
            prefix, suffix = __name__.rsplit(".", 1)

        try:
            register_tools(server_registry, prefix, {suffix: "demo"}, {})
        finally:
            # Only drop the alias we created — never this module's own entry.
            if aliased:
                sys.modules.pop(f"{prefix}.{suffix}", None)

        tool_registry = server_registry.get("demo")
        assert tool_registry is not None
        assert tool_registry.get_tool("echo") is not None
        assert tool_registry.get_tool("ping") is not None

    def test_failing_module_does_not_abort_other_modules(self, caplog):
        """One module raising during registration must not prevent the
        remaining modules from being registered."""
        import logging

        def broken_tool(server, **tool_kwargs):
            raise RuntimeError("boom: broken tool registration")

        # Mimic a @register_tool wrapper: is_register_tool() identifies tools
        # by their is_mcp_tool attribute, and the module filter requires the
        # wrapper's __module__ to equal the module's canonical __name__.
        broken_tool.is_mcp_tool = True
        broken_tool.__module__ = "_mdtr_failing_module"

        failing = types.ModuleType("_mdtr_failing_module")
        failing.broken_tool = broken_tool
        sys.modules["_mdtr_failing_module"] = failing

        prefix = "_mdtr_alias_pkg3"
        sys.modules[f"{prefix}.good"] = sys.modules[__name__]
        sys.modules[f"{prefix}.bad"] = failing
        try:
            server_registry = McpServerRegistry()
            register_tools(
                server_registry,
                prefix,
                {"bad": "server_bad", "good": "server_good"},
                {},
            )
        finally:
            sys.modules.pop(f"{prefix}.good", None)
            sys.modules.pop(f"{prefix}.bad", None)
            del sys.modules["_mdtr_failing_module"]

        # The good module was still registered despite the bad one failing.
        server_good = server_registry.get("server_good")
        assert server_good is not None
        assert len(server_good) == 2
        assert server_good.get_tool("echo") is not None
        assert server_good.get_tool("ping") is not None

        # The failure was logged at ERROR with the module name.
        error_records = [
            r
            for r in caplog.records
            if r.levelno == logging.ERROR and "_mdtr_alias_pkg3.bad" in r.getMessage()
        ]
        assert len(error_records) == 1

    def test_multiple_suffixes_map_to_distinct_servers(self):
        server_registry = McpServerRegistry()
        prefix = "_mdtr_alias_pkg2"
        sys.modules[f"{prefix}.mod_a"] = sys.modules[__name__]
        try:
            register_tools(server_registry, prefix, {"mod_a": "server_a"}, {})
        finally:
            sys.modules.pop(f"{prefix}.mod_a", None)

        server_a = server_registry.get("server_a")
        assert server_a is not None
        assert len(server_a) == 2
        assert server_registry.get("server_b") is None
