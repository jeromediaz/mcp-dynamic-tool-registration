"""Tests for mcp_dynamic_tool_registration.elicitation.confirm_destructive."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_dynamic_tool_registration.elicitation import (
    _ELICITATION_TIMEOUT_SECONDS,
    DeclinedError,
    ElicitationNotSupportedError,
    confirm_destructive,
)


def _session(*, supports_elicitation: bool = True, action: str = "accept") -> MagicMock:
    session = MagicMock()
    session.check_client_capability.return_value = supports_elicitation
    session.elicit = AsyncMock(return_value=MagicMock(action=action))
    return session


class TestConfirmDestructive:
    def test_accept_returns_true(self):
        session = _session(action="accept")
        result = asyncio.run(confirm_destructive(session, "Delete this?"))
        assert result is True

    def test_decline_returns_false(self):
        session = _session(action="decline")
        result = asyncio.run(confirm_destructive(session, "Delete this?"))
        assert result is False

    def test_cancel_returns_false(self):
        session = _session(action="cancel")
        result = asyncio.run(confirm_destructive(session, "Delete this?"))
        assert result is False

    def test_unsupported_capability_raises_and_never_elicits(self):
        session = _session(supports_elicitation=False)
        with pytest.raises(ElicitationNotSupportedError):
            asyncio.run(confirm_destructive(session, "Delete this?"))
        session.elicit.assert_not_called()

    def test_passes_message_and_empty_schema_to_elicit(self):
        session = _session(action="accept")
        asyncio.run(confirm_destructive(session, "Delete note 'Foo'?"))
        session.elicit.assert_called_once_with(
            message="Delete note 'Foo'?",
            requestedSchema={"type": "object", "properties": {}},
        )

    def test_checks_elicitation_capability_specifically(self):
        session = _session(action="accept")
        asyncio.run(confirm_destructive(session, "Delete this?"))
        (capability,), _kwargs = session.check_client_capability.call_args
        assert capability.elicitation is not None

    def test_hung_client_times_out_and_raises_declined(self, monkeypatch):
        monkeypatch.setattr(
            "mcp_dynamic_tool_registration.elicitation._ELICITATION_TIMEOUT_SECONDS",
            0.05,
        )
        session = _session(action="accept")

        async def _hang(**_kwargs: Any) -> Any:
            await asyncio.sleep(10)

        session.elicit = AsyncMock(side_effect=_hang)
        with pytest.raises(DeclinedError, match="timed out"):
            asyncio.run(confirm_destructive(session, "Delete this?"))

    def test_elicit_is_bounded_by_named_timeout_constant(self):
        assert _ELICITATION_TIMEOUT_SECONDS > 0
