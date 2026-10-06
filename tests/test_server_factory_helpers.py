"""Tests for mcp_dynamic_tool_registration.server_factory helpers."""

from __future__ import annotations

import asyncio
import json
import logging

import pydantic
from mcp import types
from pydantic import BaseModel

from mcp_dynamic_tool_registration.server_factory import (
    _MAX_ERROR_DETAIL_CHARS,
    DEFAULT_ARGUMENT_LIMITS,
    ArgumentLimits,
    AsgiApp,
    _extract_bearer_token,
    _extract_header,
    _validate_arguments,
    coerce_json_strings,
    coerce_json_strings_with_schema,
    usage_error_result,
)

LOGGER_NAME = "mcp_dynamic_tool_registration.server_factory"


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


class TestSchemaAwareCoercion:
    """Unit tests for coerce_json_strings_with_schema — coercion guided by
    the tool's input schema: JSON-looking strings are decoded only for
    fields whose declared type admits an object or array."""

    def test_scalar_string_field_starting_with_brace_is_untouched(self):
        from pydantic import BaseModel

        class In(BaseModel):
            body: str

        result = coerce_json_strings_with_schema({"body": '{"a": 1}'}, In)
        assert result["body"] == '{"a": 1}'

    def test_scalar_str_field_with_invalid_json_unchanged_and_validates(self):
        from pydantic import BaseModel

        class In(BaseModel):
            body: str

        result = coerce_json_strings_with_schema({"body": "[not valid json"}, In)
        assert result["body"] == "[not valid json"
        assert In.model_validate(result).body == "[not valid json"

    def test_container_field_double_encoded_is_still_coerced(self):
        from pydantic import BaseModel

        class In(BaseModel):
            tags: list[str] | None = None

        result = coerce_json_strings_with_schema({"tags": '["a", "b"]'}, In)
        assert result["tags"] == ["a", "b"]

    def test_optional_scalar_union_field_is_not_coerced(self):
        from pydantic import BaseModel

        class In(BaseModel):
            note_type: str | None = None

        result = coerce_json_strings_with_schema({"note_type": '{"a": 1}'}, In)
        assert result["note_type"] == '{"a": 1}'
        assert In.model_validate(result).note_type == '{"a": 1}'

    def test_int_list_string_under_scalar_field_is_not_mangled(self):
        from pydantic import BaseModel

        class In(BaseModel):
            note_id: str

        result = coerce_json_strings_with_schema({"note_id": "[42]"}, In)
        assert result == {"note_id": "[42]"}

    def test_nested_model_string_fields_not_coerced_but_container_slots_are(self):
        from pydantic import BaseModel

        class RelationEntry(BaseModel):
            predicate: str
            target_id: str

        class In(BaseModel):
            relations: list[RelationEntry] | None = None

        # Case 1: already-valid list — the scalar sub-field target_id must
        # not be rewritten even though its value looks like JSON.
        result = coerce_json_strings_with_schema(
            {"relations": [{"predicate": "references", "target_id": '["a"]'}]}, In
        )
        assert result["relations"] == [
            {"predicate": "references", "target_id": '["a"]'}
        ]

        # Case 2: the whole list is double-encoded — it is parsed, but the
        # inner scalar target_id stays a literal string.
        encoded = json.dumps([{"predicate": "references", "target_id": '["a"]'}])
        result = coerce_json_strings_with_schema({"relations": encoded}, In)
        assert result["relations"] == [
            {"predicate": "references", "target_id": '["a"]'}
        ]

    def test_scalar_field_json_scalar_string_unchanged(self):
        from pydantic import BaseModel

        class In(BaseModel):
            a: str

        result = coerce_json_strings_with_schema({"a": "42"}, In)
        assert result["a"] == "42"

    def test_no_schema_falls_back_to_legacy_walker(self):
        assert coerce_json_strings_with_schema({"a": "[1, 2]"}, None) == {"a": [1, 2]}

    def test_alias_field_is_coerced_per_its_own_type(self):
        from pydantic import BaseModel, ConfigDict, Field

        class In(BaseModel):
            model_config = ConfigDict(populate_by_name=True)
            tags: list[str] = Field(alias="tagsAlias")

        result = coerce_json_strings_with_schema({"tagsAlias": '["x"]'}, In)
        assert result["tagsAlias"] == ["x"]

    def test_any_typed_field_is_coerced(self):
        from typing import Any

        from pydantic import BaseModel

        class In(BaseModel):
            payload: Any = None

        result = coerce_json_strings_with_schema({"payload": '{"a":1}'}, In)
        assert result["payload"] == {"a": 1}


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

    def test_oversized_detail_is_clamped(self):
        """An over-long detail (e.g. one carrying error internals) must be
        truncated to the client-visible cap — never streamed to the client
        in full."""
        exc = ValueError("boom")
        exc.detail = "d" * (_MAX_ERROR_DETAIL_CHARS + 1000)  # type: ignore[attr-defined]
        result = usage_error_result("x", exc)
        text = result.content[0].text
        assert text.startswith("Tool 'x' error: ")
        assert len(text) == len("Tool 'x' error: ") + _MAX_ERROR_DETAIL_CHARS

    def test_detail_at_exactly_the_limit_is_not_truncated(self):
        exc = ValueError("boom")
        exc.detail = "d" * _MAX_ERROR_DETAIL_CHARS  # type: ignore[attr-defined]
        result = usage_error_result("x", exc)
        assert (
            result.content[0].text == "Tool 'x' error: " + "d" * _MAX_ERROR_DETAIL_CHARS
        )

    def test_logs_full_detail_with_traceback_at_warning(self, caplog):
        """The message (full untruncated detail) and the traceback go to
        the log even though the client only receives the clamp — the
        server-side/client-side detail split this function exists for."""
        exc = ValueError("secret internals " * 500)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            usage_error_result("x", exc)
        # The 4k client cap is far shorter than this message, so a log line
        # containing all of it can only be the full-detail server-side copy.
        assert len(str(exc)) > _MAX_ERROR_DETAIL_CHARS
        assert caplog.text.count("secret internals") > _MAX_ERROR_DETAIL_CHARS // 17
        (record,) = [
            r
            for r in caplog.records
            if r.name == "mcp_dynamic_tool_registration.server_factory"
        ]
        assert record.levelno == logging.WARNING
        assert record.exc_info is not None
        assert record.exc_info[1] is exc


class _EchoInput(BaseModel):
    text: str


class TestValidateArguments:
    """_validate_arguments: a mistyped field carries the exception object
    itself (so the dispatch renders a client-safe summary and logs the full
    detail), unknown-key and bounds failures stay plain strings, and
    argument bounds run last."""

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

    def test_pydantic_failure_returns_the_validation_error(self):
        """A shape failure must surface as the ``ValidationError`` itself,
        not its rendered text: the rendered form embeds the caller's input
        values and model internals, which the client is not supposed to
        see."""
        error = _validate_arguments(_EchoInput, {})
        assert isinstance(error, pydantic.ValidationError)
        assert any(err["loc"] == ("text",) for err in error.errors())

    def test_shape_checks_win_over_bounds(self):
        # A bad shape AND an oversized value: the shape error is reported.
        error = _validate_arguments(
            _EchoInput, {"text": 123}, ArgumentLimits(max_string_length=1)
        )
        assert isinstance(error, pydantic.ValidationError)
        assert any(err["type"] == "string_type" for err in error.errors())
