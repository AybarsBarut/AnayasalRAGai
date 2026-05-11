from __future__ import annotations

from typing import Any

from fastapi import status


class ConstitutionalError(Exception):
    """Application-level errors raised while processing constitutional queries."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "constitutional_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(ConstitutionalError):
    """Input validation errors."""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "validation_error"


class SecurityError(ConstitutionalError):
    """Security policy violations."""

    status_code = status.HTTP_403_FORBIDDEN
    code = "security_error"


class AuthenticationError(SecurityError):
    """Missing or invalid API key."""

    status_code = status.HTTP_401_UNAUTHORIZED
    code = "authentication_error"


def error_payload(
    *,
    code: str,
    message: str,
    details: dict[str, Any] | list[Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }
    if request_id:
        payload["request_id"] = request_id
    return payload
