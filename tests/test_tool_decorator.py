"""Tests for @register_tool against a synthetic module of decorated functions."""

from __future__ import annotations

from pydantic import BaseModel

from mcp_dynamic_tool_registration.tool_decorator import is_register_tool, register_tool

# ---------------------------------------------------------------------------
# Synthetic module of @register_tool-decorated functions
# ---------------------------------------------------------------------------


class _ItemSchema(BaseModel):
    item_id: str


@register_tool("list_items", description="List items.", read_only_hint=True)
def list_items_tool(app_context=None):
    """docstring not used since description was given explicitly"""
    return {"items": [], "app_context": app_context}


@register_tool(input_schema=_ItemSchema)
def get_item():
    """Fetch a single item."""
    return {"item": None}


@register_tool("disabled_tool", enabled=False)
def disabled_tool():  # pragma: no cover
    return {}


@register_tool("conditional_tool", enabled=lambda **p: p.get("flag", False))
def conditional_tool():
    return {"ok": True}


@register_tool("no_param_tool", enabled=lambda: True)
def no_param_tool():
    return {"ok": True}


@register_tool("subset_tool", enabled=lambda flag: flag)
def subset_tool():
    return {"ok": True}


@register_tool("versioned_tool")
def versioned_tool(*, version: int = 1, **kwargs):
    return {"version": version}


@register_tool("async_tool")
async def async_tool_impl(*, version: int = 1, **kwargs):
    return {"version": version}


@register_tool("destructive_tool", destructive_hint=True, idempotent_hint=True)
def destructive_tool():
    return {}


def plain_function():  # not decorated
    return None


# ---------------------------------------------------------------------------
# Fake server collecting add_tool() calls
# ---------------------------------------------------------------------------


class _FakeServer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add_tool(self, **kwargs):
        self.calls.append(kwargs)


# ---------------------------------------------------------------------------
# Tests: decorator itself
# ---------------------------------------------------------------------------


class TestRegisterToolDecorator:
    def test_marks_is_mcp_tool(self):
        assert list_items_tool.is_mcp_tool is True

    def test_is_register_tool_predicate_true(self):
        assert is_register_tool(list_items_tool)

    def test_is_register_tool_predicate_false_for_plain_func(self):
        assert not is_register_tool(plain_function)

    def test_tool_meta_shape(self):
        meta = list_items_tool._tool_meta
        assert meta["name"] == "list_items"
        assert meta["description"] == "List items."
        assert meta["annotations"] == {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        }

    def test_default_name_from_function_name(self):
        assert get_item._tool_meta["name"] == "get_item"

    def test_default_description_from_docstring(self):
        assert get_item._tool_meta["description"] == "Fetch a single item."

    def test_input_schema_stored(self):
        assert get_item._tool_meta["input_schema"] is _ItemSchema

    def test_destructive_and_idempotent_annotations(self):
        annotations = destructive_tool._tool_meta["annotations"]
        assert annotations["destructiveHint"] is True
        assert annotations["idempotentHint"] is True


# ---------------------------------------------------------------------------
# Tests: wrapper(server, **params) registration behavior
# ---------------------------------------------------------------------------


class TestRegisterToolWrapper:
    def test_registers_on_server(self):
        server = _FakeServer()
        list_items_tool(server)
        assert len(server.calls) == 1
        call = server.calls[0]
        assert call["name"] == "list_items"
        assert call["description"] == "List items."
        assert callable(call["handler"])

    def test_disabled_tool_not_registered(self):
        server = _FakeServer()
        disabled_tool(server)
        assert server.calls == []

    def test_conditional_tool_enabled(self):
        server = _FakeServer()
        conditional_tool(server, flag=True)
        assert len(server.calls) == 1

    def test_conditional_tool_disabled(self):
        server = _FakeServer()
        conditional_tool(server, flag=False)
        assert server.calls == []

    def test_params_injected_as_defaults(self):
        server = _FakeServer()
        versioned_tool(server, version=42)
        handler = server.calls[0]["handler"]
        assert handler() == {"version": 42}

    def test_default_params_without_injection(self):
        server = _FakeServer()
        versioned_tool(server)
        handler = server.calls[0]["handler"]
        assert handler() == {"version": 1}

    def test_var_keyword_stripped_from_signature(self):
        import inspect

        server = _FakeServer()
        versioned_tool(server, version=1)
        handler = server.calls[0]["handler"]
        sig = inspect.signature(handler)
        assert not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

    async def _call_async(self, handler):
        return await handler()

    def test_async_handler_still_awaitable(self):
        import asyncio

        server = _FakeServer()
        async_tool_impl(server, version=7)
        handler = server.calls[0]["handler"]
        result = asyncio.run(self._call_async(handler))
        assert result == {"version": 7}

    def test_unrecognized_param_silently_dropped(self):
        """Params not in the function signature (and no **kwargs) don't raise."""
        server = _FakeServer()
        get_item(server, app_context=object(), unrelated="x")
        assert len(server.calls) == 1

    def test_enabled_callback_no_params(self):
        """An enabled callback with no params works even when params are passed."""
        server = _FakeServer()
        no_param_tool(server, flag=True, app_context=object())
        assert len(server.calls) == 1

    def test_enabled_callback_subset_params(self):
        """An enabled callback receiving only a subset of params works."""
        server = _FakeServer()
        subset_tool(server, flag=True, other_param="x")
        assert len(server.calls) == 1

    def test_enabled_callback_subset_params_disabled(self):
        server = _FakeServer()
        subset_tool(server, flag=False, other_param="x")
        assert server.calls == []
