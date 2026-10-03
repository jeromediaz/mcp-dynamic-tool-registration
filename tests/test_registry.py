"""Tests for mcp_dynamic_tool_registration.registry."""

from __future__ import annotations

import inspect

import pytest

from mcp_dynamic_tool_registration.registry import (
    McpServerRegistry,
    ToolRegistry,
    resolve_server_name,
)

# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_add_and_get_tool(self):
        registry = ToolRegistry("demo")

        def handler() -> None: ...

        registry.add_tool(
            name="list_notes",
            description="List notes",
            handler=handler,
            annotations={"readOnlyHint": True},
        )
        spec = registry.get_tool("list_notes")
        assert spec is not None
        assert spec.name == "list_notes"
        assert spec.handler is handler
        assert spec.annotations == {"readOnlyHint": True}

    def test_get_missing_tool_returns_none(self):
        registry = ToolRegistry("demo")
        assert registry.get_tool("nope") is None

    def test_list_tools_and_len(self):
        registry = ToolRegistry("demo")
        registry.add_tool(name="a", description="", handler=lambda: None)
        registry.add_tool(name="b", description="", handler=lambda: None)
        assert len(registry) == 2
        assert {spec.name for spec in registry.list_tools()} == {"a", "b"}

    def test_handler_signature_reflects_handler_parameters(self):
        registry = ToolRegistry("demo")

        def handler(note_id: str, limit: int = 10, *, context=None) -> dict:
            return {"note_id": note_id, "limit": limit, "context": context}

        registry.add_tool(
            name="with_args",
            description="Has parameters",
            handler=handler,
        )
        spec = registry.get_tool("with_args")
        assert spec is not None
        params = spec.handler_signature.parameters
        assert list(params) == ["note_id", "limit", "context"]
        # The module uses ``from __future__ import annotations``, so the
        # annotation is the string form rather than the ``str`` class itself.
        assert params["note_id"].annotation == "str"
        assert params["limit"].default == 10
        assert params["context"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_add_tool_extra_captures_audit_model(self):
        registry = ToolRegistry("demo")
        sentinel = object()

        registry.add_tool(
            name="audited",
            description="Carries an audit model",
            handler=lambda: None,
            audit_model=sentinel,
        )
        spec = registry.get_tool("audited")
        assert spec is not None
        assert spec.extra["audit_model"] is sentinel


# ---------------------------------------------------------------------------
# resolve_server_name
# ---------------------------------------------------------------------------


class TestResolveServerName:
    def test_bare_string(self):
        assert resolve_server_name("demo") == "demo"

    def test_mapping_with_server_name(self):
        assert resolve_server_name({"server_name": "demo"}) == "demo"

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            resolve_server_name({"nope": "x"})

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            resolve_server_name(123)


# ---------------------------------------------------------------------------
# McpServerRegistry
# ---------------------------------------------------------------------------


class TestMcpServerRegistry:
    def test_get_or_create_returns_same_instance(self):
        registry = McpServerRegistry()
        first = registry.get_or_create("demo")
        second = registry.get_or_create("demo")
        assert first is second

    def test_get_or_create_different_names(self):
        registry = McpServerRegistry()
        a = registry.get_or_create("demo")
        b = registry.get_or_create("other")
        assert a is not b
        assert set(registry.names()) == {"demo", "other"}

    def test_get_missing_returns_none(self):
        registry = McpServerRegistry()
        assert registry.get("nope") is None
