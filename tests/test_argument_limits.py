"""Unit tests for argument_limits — the size layer on raw tool-call
arguments (per-string cap, per-list cap, total-characters budget,
per-field caps keyed by input-schema class name)."""

from __future__ import annotations

import dataclasses

import pytest

from mcp_dynamic_tool_registration.argument_limits import (
    DEFAULT_ARGUMENT_LIMITS,
    MAX_ARG_LIST_LENGTH,
    MAX_ARG_STRING_LENGTH,
    MAX_ARG_TOTAL_CHARS,
    ArgumentLimits,
    validate_argument_bounds,
)


class TestDefaults:
    def test_defaults_are_transport_safety_sized(self):
        assert DEFAULT_ARGUMENT_LIMITS.max_string_length == MAX_ARG_STRING_LENGTH
        assert DEFAULT_ARGUMENT_LIMITS.max_list_length == MAX_ARG_LIST_LENGTH
        assert DEFAULT_ARGUMENT_LIMITS.max_total_chars == MAX_ARG_TOTAL_CHARS
        assert DEFAULT_ARGUMENT_LIMITS.field_max_length == {}

    def test_limits_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_ARGUMENT_LIMITS.max_string_length = 1  # type: ignore[misc]


class TestStringCap:
    def test_string_over_max_is_rejected_naming_path_and_limit(self):
        limits = ArgumentLimits(max_string_length=10)
        error = validate_argument_bounds({"body": "x" * 11}, limits)
        assert error is not None
        assert "body" in error
        assert "11 characters" in error
        assert "limit is 10" in error

    def test_string_at_max_is_within_bounds(self):
        limits = ArgumentLimits(max_string_length=10)
        assert validate_argument_bounds({"body": "x" * 10}, limits) is None

    def test_deep_string_uses_full_argument_path(self):
        limits = ArgumentLimits(max_string_length=3)
        error = validate_argument_bounds({"a": {"b": [{"c": "xxxx"}]}}, limits)
        assert error is not None
        assert "a.b[0].c" in error

    def test_string_in_a_list_element_is_capped(self):
        limits = ArgumentLimits(max_string_length=4)
        error = validate_argument_bounds({"tags": ["ok", "toolong"]}, limits)
        assert error is not None
        assert "tags[1]" in error


class TestListCap:
    def test_list_over_max_is_rejected(self):
        limits = ArgumentLimits(max_list_length=5)
        error = validate_argument_bounds({"notes": list(range(6))}, limits)
        assert error is not None
        assert "notes" in error
        assert "6 items" in error
        assert "limit is 5" in error

    def test_list_at_max_is_within_bounds(self):
        limits = ArgumentLimits(max_list_length=5)
        assert validate_argument_bounds({"notes": list(range(5))}, limits) is None

    def test_nested_list_is_capped_too(self):
        limits = ArgumentLimits(max_list_length=2)
        error = validate_argument_bounds({"a": {"b": [1, 2, 3]}}, limits)
        assert error is not None
        assert "a.b" in error


class TestTotalCharsBudget:
    def test_total_chars_over_budget_is_rejected(self):
        # Each string passes the per-string cap; the whole payload doesn't.
        limits = ArgumentLimits(max_string_length=100, max_total_chars=250)
        error = validate_argument_bounds(
            {"parts": ["x" * 100, "y" * 100, "z" * 100]}, limits
        )
        assert error is not None
        assert "total argument-size limit" in error
        assert "250" in error

    def test_total_chars_at_budget_is_within_bounds(self):
        limits = ArgumentLimits(max_string_length=100, max_total_chars=200)
        assert (
            validate_argument_bounds({"parts": ["x" * 100, "y" * 100]}, limits) is None
        )


class TestFieldCaps:
    def test_field_cap_beats_generic_cap(self):
        limits = ArgumentLimits(
            max_string_length=1_000,
            field_max_length={"CreateNoteInput": {"body": 10}},
        )
        assert (
            validate_argument_bounds(
                {"body": "x" * 10}, limits, schema_name="CreateNoteInput"
            )
            is None
        )
        error = validate_argument_bounds(
            {"body": "x" * 11}, limits, schema_name="CreateNoteInput"
        )
        assert error is not None
        # The FIELD cap (10) applies, not the generic one (1000).
        assert "limit is 10" in error

    def test_field_caps_are_keyed_by_schema_name(self):
        limits = ArgumentLimits(field_max_length={"OtherInput": {"body": 4}})
        # The cap is declared for a different tool's schema — this call's
        # 50-char body is only bound by the generic cap.
        assert (
            validate_argument_bounds(
                {"body": "x" * 50}, limits, schema_name="CreateNoteInput"
            )
            is None
        )

    def test_dollar_cascades_cap_into_batch_list_items(self):
        # A batch's item model is nested, so its caps are keyed on the
        # top-level list field with the `$` pseudo-field: every string in
        # every item is capped, and the message names the exact item field.
        limits = ArgumentLimits(
            max_string_length=10_000,
            field_max_length={"BulkCreateNotesInput": {"notes": 100}},
        )
        payload = {
            "notes": [
                {"title": "a", "body": "x" * 100},
                {"title": "b", "body": "y" * 101},
            ]
        }
        error = validate_argument_bounds(
            payload, limits, schema_name="BulkCreateNotesInput"
        )
        assert error is not None
        assert "notes[1].body" in error
        assert "limit is 100" in error

    def test_dollar_cascade_does_not_leak_to_other_fields(self):
        limits = ArgumentLimits(
            max_string_length=1_000,
            field_max_length={"BulkCreateNotesInput": {"notes": 10}},
        )
        # `title` is a top-level string of the same call, not a batch item.
        assert (
            validate_argument_bounds(
                {"notes": ["x"], "title": "y" * 500},
                limits,
                schema_name="BulkCreateNotesInput",
            )
            is None
        )

    def test_explicit_dollar_pseudo_field_cascades_into_a_field_subtree(self):
        # `$` is the cascade's own key in a caps map: at the schema level it
        # caps every string under every field, and it propagates down any
        # mapping-valued field (its lone-cascade child map carries it to
        # every depth), while a key's explicit cap takes precedence.
        limits = ArgumentLimits(
            max_string_length=1_000,
            field_max_length={"X": {"$": 7, "filters": 5}},
        )
        payload = {"filters": "abc", "meta": {"deep": {"leaf": "abcdefg"}}}
        assert validate_argument_bounds(payload, limits, schema_name="X") is None
        # The cascaded cap (7) fires for a string nested under a field
        # without its own cap — not the generic cap (1000).
        error = validate_argument_bounds(
            {"filters": "abc", "meta": {"deep": {"leaf": "abcdefgh"}}},
            limits,
            schema_name="X",
        )
        assert error is not None
        assert "meta.deep.leaf" in error
        assert "limit is 7" in error
        # ``filters``'s explicit cap (5) takes precedence over the
        # cascade (7), so it is the limit named in its own error.
        error = validate_argument_bounds({"filters": "abcdef"}, limits, schema_name="X")
        assert error is not None
        assert "limit is 5" in error

    def test_dollar_cascade_uses_min_with_a_parent_field_cap(self):
        # The `$` pseudo-field also governs a field's *list* elements, and
        # the cap in force is min(cascade, the parent field's cap).
        limits = ArgumentLimits(
            max_string_length=1_000,
            field_max_length={"X": {"tags": 20, "$": 3}},
        )
        assert (
            validate_argument_bounds({"tags": ["abc"]}, limits, schema_name="X") is None
        )
        error = validate_argument_bounds({"tags": ["abcd"]}, limits, schema_name="X")
        assert error is not None
        assert "tags[0]" in error
        # min(3, 20): the cascade wins because it is the tighter cap.
        assert "limit is 3" in error

    def test_nested_field_key_caps_a_nested_object_field(self):
        # `NoteUpdateFields` is nested under the top-level `updates` key;
        # its cap is keyed on the parent schema and the parent key.
        limits = ArgumentLimits(
            max_string_length=10_000,
            field_max_length={"UpdateNoteInput": {"updates": 10}},
        )
        assert (
            validate_argument_bounds(
                {"note_id": "n", "updates": {"body": "x" * 10}},
                limits,
                schema_name="UpdateNoteInput",
            )
            is None
        )
        error = validate_argument_bounds(
            {"note_id": "n", "updates": {"body": "x" * 11}},
            limits,
            schema_name="UpdateNoteInput",
        )
        assert error is not None
        assert "updates.body" in error
        assert "limit is 10" in error

    def test_per_field_cap_also_bounds_the_total_chars_budget(self):
        # Cascade must not make total budget unenforceable: the per-item
        # cap (5) x 3 items is under the generic cap but over the budget.
        limits = ArgumentLimits(
            max_string_length=1_000,
            max_total_chars=10,
            field_max_length={"BulkCreateNotesInput": {"notes": 5}},
        )
        error = validate_argument_bounds(
            {"notes": [{"body": "x" * 5}, {"body": "y" * 5}, {"body": "z" * 5}]},
            limits,
            schema_name="BulkCreateNotesInput",
        )
        assert error is not None
        assert "total argument-size limit" in error


class TestTraversalSafety:
    def test_deeply_nested_explosion_is_rejected_without_recursion_error(self):
        # A client can build a list nested thousands deep cheaply, with
        # the violation at the bottom (visited last by a LIFO walk); the
        # iterative traversal must reject it by size, not RecursionError.
        payload: dict = {"leaf": "x" * 10}
        for _ in range(50_000):
            payload = {"a": payload}
        limits = ArgumentLimits(max_string_length=5)
        error = validate_argument_bounds(payload, limits)
        assert error is not None
        assert "length limit" in error

    def test_deeply_nested_list_is_traversed_without_recursion_error(self):
        # Deep nesting alone must not break the walk (a deep list is a
        # DFS path, not a traversal explosion — aliased containers are
        # the exponential case, and rejected outright).
        payload: list | dict = {"x": 1}
        for _ in range(50_000):
            payload = [payload]
        assert validate_argument_bounds({"a": payload}, DEFAULT_ARGUMENT_LIMITS) is None

    def test_aliased_container_is_rejected(self):
        shared = {"body": "x"}
        error = validate_argument_bounds(
            {"a": shared, "b": shared, "c": 1}, DEFAULT_ARGUMENT_LIMITS
        )
        assert error is not None
        assert "aliased" in error

    def test_non_container_values_pass_through(self):
        assert validate_argument_bounds(42, DEFAULT_ARGUMENT_LIMITS) is None
        assert validate_argument_bounds("x", DEFAULT_ARGUMENT_LIMITS) is None
        assert validate_argument_bounds(None, DEFAULT_ARGUMENT_LIMITS) is None

    def test_within_default_limits_returns_none(self):
        payload = {
            "title": "note",
            "body": "x" * 1000,
            "tags": ["a", "b"],
            "nested": {"lists": [[1, 2], [3]]},
        }
        assert validate_argument_bounds(payload, DEFAULT_ARGUMENT_LIMITS) is None
