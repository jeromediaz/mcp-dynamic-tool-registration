"""Elicitation confirmation wiring for destructive tools.

A destructive tool calls ``confirm_destructive`` before performing the
irreversible action. ``mcp_session`` is the MCP SDK's per-request
``ServerSession`` — threaded to a tool function the same way ``context`` is
(an explicit parameter, injected per call by ``server_factory``'s
dispatch, never a registration-time static default), which also makes it
trivial to fake in unit tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp import types

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

# Maximum time (seconds) to wait for the connected MCP client to answer an
# elicitation confirmation prompt. A hung or unresponsive client must not
# block a destructive tool call indefinitely; on timeout the call fails
# closed (the confirmation is treated as declined).
_ELICITATION_TIMEOUT_SECONDS = 60.0

_ELICITATION_CAPABILITY = types.ClientCapabilities(
    elicitation=types.ElicitationCapability()
)


class ElicitationNotSupportedError(Exception):
    """Raised when a destructive tool needs confirmation but the connected
    MCP client didn't declare elicitation support at ``initialize`` — fails
    closed rather than silently proceeding without confirmation."""


class DeclinedError(Exception):
    """Raised when the user explicitly declines (or cancels) an elicitation
    confirmation for a destructive tool call, or when the elicitation
    request times out waiting for the client to respond. Distinct from an
    ACL denial — a host audit hook may record this as "declined", not
    "denied"."""


async def confirm_destructive(mcp_session: Any, message: str) -> bool:
    """Ask the connected MCP client to confirm a destructive action via
    ``elicitation/create``. Returns ``True`` only on an explicit "accept" —
    both "decline" and "cancel" return ``False``.

    Raises ``ElicitationNotSupportedError`` if the client didn't negotiate
    elicitation support at ``initialize``, rather than silently proceeding
    without confirmation.

    Raises ``DeclinedError`` if the client does not answer within
    ``_ELICITATION_TIMEOUT_SECONDS`` — a hung or unresponsive client fails
    closed, as if the user had declined, rather than blocking the
    destructive tool call indefinitely.
    """
    if not mcp_session.check_client_capability(_ELICITATION_CAPABILITY):
        raise ElicitationNotSupportedError(
            "Connected MCP client does not support elicitation; refusing to "
            "perform a destructive action without confirmation"
        )

    try:
        result: types.ElicitResult = await asyncio.wait_for(
            mcp_session.elicit(message=message, requestedSchema=_EMPTY_SCHEMA),
            timeout=_ELICITATION_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        raise DeclinedError(
            f"Elicitation confirmation timed out after "
            f"{_ELICITATION_TIMEOUT_SECONDS:g}s waiting for the client; "
            "refusing to perform a destructive action without confirmation"
        ) from None
    return result.action == "accept"
