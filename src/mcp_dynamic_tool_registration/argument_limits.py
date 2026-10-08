"""Size bounds for raw tool-call arguments.

Pydantic field validation answers "is this the right *shape*?" but says
nothing about size: a tool whose schema is ``body: str`` accepts a 500 MB
string, ``notes: list[...]`` a million items, and no schema bounds the
*total* payload.  This module adds that missing size layer:
:func:`validate_argument_bounds` walks raw (already-coerced) ``call_tool``
arguments and rejects the first value that exceeds a limit, naming the
exact limit and the argument path (e.g. ``notes[0].body``) in the message.

The defaults are library-wide and sized for transport safety only, so the
library never silently blocks legitimate content a host would accept;
hosts opt into tighter per-field caps through
:attr:`ArgumentLimits.field_max_length` (mirroring their REST-side quota
assumptions).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

MAX_ARG_STRING_LENGTH = 5_000_000
"""Default cap on any single string value in tool-call arguments
(transport-safety sized: large enough that no legitimate tool payload is
blocked by the library itself)."""

MAX_ARG_LIST_LENGTH = 1000
"""Default cap on any single list value in tool-call arguments."""

MAX_ARG_TOTAL_CHARS = 20_000_000
"""Default cap on the total accumulated string characters of one call's
arguments (a payload this large is a transport problem before it is a
domain one)."""

_CASCADE = "$"
"""Pseudo-field in :attr:`ArgumentLimits.field_max_length` naming a
field's *elements*: the cap cascades into every string under the field,
at any nesting depth (``{"BulkCreateNotesInput": {"notes": 100}}`` bounds
every string inside every item of a ``bulk_create_notes`` batch)."""


@dataclass(frozen=True)
class ArgumentLimits:
    """Size bounds for raw tool-call arguments (see
    :func:`validate_argument_bounds`).

    Attributes:
        max_string_length: Cap on any individual ``str`` value, unless a
            per-field cap applies.
        max_list_length: Cap on any individual ``list`` value, at any
            nesting depth.
        max_total_chars: Cap on the total string characters of the whole
            payload.
        field_max_length: Per-field string-length caps keyed by
            ``tool input-schema class name -> field name -> cap``, where
            the field names follow the schema's own JSON shape: top-level
            argument keys, and the ``$`` pseudo-field naming a field's
            elements (their own fields inherit the same cap, so one entry
            bounds every string inside a batch). A cap declared for a key
            takes precedence over the generic string cap, and a cascaded
            cap applies to everything under its key — bounded by ``min``
            with the cap above it — so nested models are bounded through
            the top-level key that holds them (their own class name is not
            visible to the raw payload walk). At the schema level ``$``
            cascades into every field's subtree: each string under a
            mapping-valued field carries the cascade to every depth, while
            a key's own cap always wins over it.
    """

    max_string_length: int = MAX_ARG_STRING_LENGTH
    max_list_length: int = MAX_ARG_LIST_LENGTH
    max_total_chars: int = MAX_ARG_TOTAL_CHARS
    field_max_length: Mapping[str, Mapping[str, int]] = field(default_factory=dict)


DEFAULT_ARGUMENT_LIMITS = ArgumentLimits()
"""Library defaults (transport safety only) — hosts inject tighter caps."""

_Frame = tuple[Any, str, Mapping[str, int] | None, int]
"""One pending traversal step: ``(value, path, caps, cap)``."""


@dataclass
class _Walk:
    """Traversal state for :func:`validate_argument_bounds`: the limits,
    the running total of string characters, and every container already
    seen (aliased payloads are rejected outright)."""

    limits: ArgumentLimits
    total_chars: int = 0
    visited: set[int] = field(default_factory=set)

    def visit(
        self,
        stack: list[_Frame],
        item: Any,
        path: str,
        caps: Mapping[str, int] | None,
        cap: int,
    ) -> str | None:
        """Check *item* against the cap in force and push its children;
        return an error message when a bound is violated."""
        if isinstance(item, str):
            return self._visit_string(item, path, cap)
        if isinstance(item, Mapping):
            return self._visit_mapping(stack, item, path, caps, cap)
        if isinstance(item, list):
            return self._visit_list(stack, item, path, caps, cap)
        return None

    def _visit_string(self, item: str, path: str, cap: int) -> str | None:
        self.total_chars += len(item)
        if len(item) > cap:
            return _length_error(path, len(item), cap)
        if self.total_chars > self.limits.max_total_chars:
            return (
                f"Argument '{path}' exceeds the total argument-size limit "
                f"({self.limits.max_total_chars} characters across all arguments)"
            )
        return None

    def _visit_mapping(
        self,
        stack: list[_Frame],
        item: Mapping[Any, Any],
        path: str,
        caps: Mapping[str, int] | None,
        cap: int,
    ) -> str | None:
        if id(item) in self.visited:
            return _aliasing_error(path)
        self.visited.add(id(item))
        cascaded: int | None = None
        if caps is not None:
            cascaded = caps.get(_CASCADE)
        child_caps: Mapping[str, int] | None = (
            {_CASCADE: cascaded} if cascaded is not None else None
        )
        for key, nested in item.items():
            if caps is not None and key in caps:
                key_cap = caps[key]
            elif cascaded is not None:
                key_cap = min(cascaded, cap)
            else:
                key_cap = cap
            stack.append((nested, _child_path(path, str(key)), child_caps, key_cap))
        return None

    def _visit_list(
        self,
        stack: list[_Frame],
        item: list[object],
        path: str,
        caps: Mapping[str, int] | None,
        cap: int,
    ) -> str | None:
        if len(item) > self.limits.max_list_length:
            return (
                f"Argument '{path}' exceeds the list-size limit: {len(item)} "
                f"items, limit is {self.limits.max_list_length}"
            )
        if id(item) in self.visited:
            return _aliasing_error(path)
        self.visited.add(id(item))
        element_cap = cap
        if caps is not None:
            cascaded = caps.get(_CASCADE)
            if cascaded is not None:
                element_cap = min(cascaded, cap)
        for index, element in enumerate(item):
            stack.append((element, f"{path}[{index}]", None, element_cap))
        return None


def validate_argument_bounds(
    value: Any,
    limits: ArgumentLimits,
    *,
    schema_name: str | None = None,
    path: str = "",
) -> str | None:
    """Check *value* (raw tool-call arguments) against *limits*; return an
    error message naming the limit and the argument path, or ``None`` when
    everything is within bounds.

    Args:
        value: The (already JSON-coerced) arguments payload to check — a
            mapping at the top level, but any JSON value is accepted.
        limits: The bounds to enforce.
        schema_name: Name of the tool's pydantic input-schema class; keys
            ``limits.field_max_length`` against it.
        path: Argument path of *value* in error messages ("" at the top).

    The traversal is iterative (an explicit work stack, not recursion), so
    a deeply nested payload — which a client can build cheaply — is
    rejected by a size limit instead of hitting ``RecursionError``
    partway through validation. A payload that places the same container
    in two places is rejected outright: aliased structures are malformed
    JSON at best and an exponential traversal bomb at worst.

    Frames are ``(value, path, caps, cap)``. ``caps`` is the cap map in
    force for a *mapping* frame: the input schema's own field caps at the
    root, ``{$: c}`` — a lone cascade — under a field that declared one,
    and nothing below a field list (``[{$: c}, cap]`` carries the cascade
    through ``cap`` instead, where it becomes each element's own cap).
    ``cap`` is the size limit in force for the value itself — a field cap,
    a cascade (then it propagates into every container under the value
    too, bounded by ``min`` with the cap above it), or the generic string
    cap at the root.
    """
    root_caps: Mapping[str, int] | None = None
    if isinstance(value, Mapping) and schema_name is not None:
        root_caps = limits.field_max_length.get(schema_name)
    walk = _Walk(limits)
    stack: list[_Frame] = [(value, path, root_caps, limits.max_string_length)]
    while stack:
        item, item_path, caps, cap = stack.pop()
        error = walk.visit(stack, item, item_path, caps, cap)
        if error is not None:
            return error
    return None


def _length_error(path: str, actual: int, limit: int) -> str:
    return (
        f"Argument '{path}' exceeds the length limit: {actual} characters, "
        f"limit is {limit}"
    )


def _aliasing_error(path: str) -> str:
    return (
        f"Argument '{path}' repeats a container already present in the "
        "payload (aliased structures are not accepted)"
    )


def _child_path(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key
