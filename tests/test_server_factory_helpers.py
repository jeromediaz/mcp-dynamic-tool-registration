"""Tests for mcp_dynamic_tool_registration.server_factory helpers."""

from __future__ import annotations

import asyncio
import json

from mcp import types
from pydantic import BaseModel

from mcp_dynamic_tool_registration.server_factory import (
    DEFAULT_ARGUMENT_LIMITS,
    ArgumentLimits,
    AsgiApp,
    _extract_bearer_token,
    _extract_header,
    _validate_arguments,
    coerce_json_strings,
    usage_error_result,
)


class TestCoerceJsonStrings:
    """Unit tests for coerce_json_strings — the generic defence against
    LLM clients that double-encode tool-call arguments (send a JSON string
    where a list/dict is expected)."""

    def test_plain_string_unchanged(self):
        assert coerce_json_strings("hello") == "hello"

    def test_empty_string_unchanged(self):
        assert coerce_json_strings("") == ""

    def test_string_with_whitespace_before_bracket_unchanged(self):
        assert coerce_json_strings("  hello") == "  hello"

    def test_string_containing_json_but_not_starting_with_bracket_unchanged(self):
        assert coerce_json_strings("value is [1,2]") == "value is [1,2]"

    def test_json_array_string_coerced_to_list(self):
        result = coerce_json_strings('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_json_object_string_coerced_to_dict(self):
        result = coerce_json_strings('{"key": "value"}')
        assert result == {"key": "value"}

    def test_nested_json_array_string_coerced_recursively(self):
        # The outer value is a JSON array containing an object whose "tags"
        # field is itself a JSON-encoded string (double-encoding).
        inner = json.dumps(["x", "y"])
        outer = json.dumps([{"tags": inner, "n": 1}])
        result = coerce_json_strings(outer)
        assert result == [{"tags": ["x", "y"], "n": 1}]

    def test_invalid_json_string_unchanged(self):
        assert coerce_json_strings("[not valid json") == "[not valid json"

    def test_json_scalar_string_not_coerced(self):
        assert coerce_json_strings('"just a string"') == '"just a string"'
        assert coerce_json_strings("42") == "42"
        assert coerce_json_strings("true") == "true"
        assert coerce_json_strings("null") == "null"

    def test_list_with_nested_strings_coerced(self):
        result = coerce_json_strings(["plain", '{"a": 1}', "[2, 3]"])
        assert result == ["plain", {"a": 1}, [2, 3]]

    def test_dict_with_nested_strings_coerced(self):
        result = coerce_json_strings({"a": "[1, 2]", "b": "plain", "c": 42})
        assert result == {"a": [1, 2], "b": "plain", "c": 42}

    def test_int_unchanged(self):
        assert coerce_json_strings(42) == 42

    def test_bool_unchanged(self):
        assert coerce_json_strings(True) is True

    def test_none_unchanged(self):
        assert coerce_json_strings(None) is None

    def test_whitespace_padded_json_array_coerced(self):
        result = coerce_json_strings("  [1, 2, 3]  ")
        assert result == [1, 2, 3]


class TestExtractBearerToken:
    def test_extracts_token_from_header(self):
        scope = {"headers": [(b"authorization", b"Bearer abc123")]}
        assert _extract_bearer_token(scope) == "abc123"

    def test_case_insensitive_bearer_prefix(self):
        scope = {"headers": [(b"authorization", b"bearer abc123")]}
        assert _extract_bearer_token(scope) == "abc123"

    def test_case_insensitive_header_name(self):
        scope = {"headers": [(b"Authorization", b"Bearer abc123")]}
        assert _extract_bearer_token(scope) == "abc123"

    def test_missing_header_returns_none(self):
        assert _extract_bearer_token({"headers": []}) is None

    def test_non_bearer_scheme_returns_none(self):
        scope = {"headers": [(b"authorization", b"Basic abc123")]}
        assert _extract_bearer_token(scope) is None

    def test_empty_token_after_scheme_returns_none(self):
        scope = {"headers": [(b"authorization", b"bearer ")]}
        assert _extract_bearer_token(scope) is None

    def test_whitespace_only_token_after_scheme_returns_none(self):
        scope = {"headers": [(b"authorization", b"Bearer   ")]}
        assert _extract_bearer_token(scope) is None


class TestExtractHeader:
    def test_case_insensitive_lookup(self):
        scope = {"headers": [(b"Mcp-Session-Id", b" s-1 ")]}
        assert _extract_header(scope, "mcp-session-id") == "s-1"

    def test_missing_header_returns_none(self):
        assert _extract_header({"headers": []}, "mcp-session-id") is None
        assert _extract_header({}, "mcp-session-id") is None

    def test_blank_value_returns_none(self):
        scope = {"headers": [(b"mcp-session-id", b"   ")]}
        assert _extract_header(scope, "mcp-session-id") is None


class TestAsgiApp:
    def test_copies_identity_from_wrapped_function(self):
        async def my_app(scope, receive, send):
            pass

        wrapped = AsgiApp(my_app)
        assert wrapped.__name__ == "my_app"
        assert wrapped.__module__ == my_app.__module__

    def test_missing_identity_falls_back_to_class(self):
        class Bare:
            async def __call__(self, scope, receive, send):
                pass

        instance = Bare()
        assert not hasattr(instance, "__name__")
        wrapped = AsgiApp(instance)
        # No __name__ on the callable -> falls back to the wrapper's class
        # name; the instance's class provides __module__.
        assert wrapped.__name__ == "AsgiApp"
        assert wrapped.__module__ == Bare.__module__

    def test_delegates_call_to_wrapped_function(self):
        calls = []

        async def fake_app(scope, receive, send):
            calls.append((scope, receive, send))

        wrapped = AsgiApp(fake_app)

        scope = {"type": "http"}

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        asyncio.run(wrapped(scope, receive, send))

        assert calls == [(scope, receive, send)]


class TestUsageErrorResult:
    def test_uses_detail_when_present(self):
        exc = ValueError("plain message")
        exc.detail = "the detail"  # type: ignore[attr-defined]
        result = usage_error_result("x", exc)
        assert result.isError is True
        assert result.content == [
            types.TextContent(type="text", text="Tool 'x' error: the detail")
        ]

    def test_falls_back_to_str_when_no_detail(self):
        result = usage_error_result("x", ValueError("plain message"))
        assert result.isError is True
        assert result.content == [
            types.TextContent(type="text", text="Tool 'x' error: plain message")
        ]

    def test_text_starts_with_tool_error_prefix(self):
        result = usage_error_result("x", RuntimeError("boom"))
        text = result.content[0].text
        assert isinstance(text, str)
        assert text.startswith("Tool 'x' error: ")


class _EchoInput(BaseModel):
    text: str


class TestValidateArguments:
    """_validate_arguments: shape checks (pydantic + unknown keys) keep
    their messages, and argument bounds run last."""

    def test_in_bounds_arguments_return_none(self):
        assert (
            _validate_arguments(_EchoInput, {"text": "hi"}, DEFAULT_ARGUMENT_LIMITS)
            is None
        )

    def test_out_of_bounds_arguments_return_a_message(self):
        error = _validate_arguments(
            _EchoInput, {"text": "x" * 11}, ArgumentLimits(max_string_length=10)
        )
        assert error is not None
        assert "limit is 10" in error

    def test_unknown_key_message_unchanged(self):
        error = _validate_arguments(_EchoInput, {"text": "hi", "nope": 1})
        assert error == "Unexpected argument(s): nope"

    def test_pydantic_message_unchanged(self):
        error = _validate_arguments(_EchoInput, {})
        assert error is not None
        assert "Field required" in error

    def test_shape_checks_win_over_bounds(self):
        # A bad shape AND an oversized value: the shape error is reported.
        error = _validate_arguments(
            _EchoInput, {"text": 123}, ArgumentLimits(max_string_length=1)
        )
        assert error is not None
        assert "string_type" in error
