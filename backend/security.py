from __future__ import annotations

from collections import defaultdict, deque
import re
import time
from typing import Deque

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.config import Settings
from backend.exceptions import AuthenticationError, SecurityError, ValidationError, error_payload


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"\s+")
PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+message\b", re.IGNORECASE),
    re.compile(r"\bonceki\s+(talimat|yonerge)leri\s+yok\s+say\b", re.IGNORECASE),
    re.compile(r"\bsistem\s+(talimat|mesaj)ini\s+(goster|acikla|yaz|ifsa)\b", re.IGNORECASE),
)


def sanitize_query(query: str, settings: Settings) -> str:
    cleaned = CONTROL_CHARS.sub(" ", query)
    cleaned = WHITESPACE.sub(" ", cleaned).strip()

    if len(cleaned) < settings.query_min_length:
        raise ValidationError(
            "Soru cok kisa.",
            {"min_length": settings.query_min_length},
        )
    if len(cleaned) > settings.query_max_length:
        raise ValidationError(
            "Soru izin verilen uzunlugu asiyor.",
            {"max_length": settings.query_max_length},
        )

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(cleaned):
            raise SecurityError("Soru guvenlik politikasi nedeniyle reddedildi.")

    return cleaned


async def verify_api_key(request: Request, settings: Settings) -> None:
    if not settings.api_key:
        return

    header_key = request.headers.get("x-api-key")
    bearer = request.headers.get("authorization", "")
    bearer_key = bearer.removeprefix("Bearer ").strip() if bearer.startswith("Bearer ") else None

    if header_key == settings.api_key or bearer_key == settings.api_key:
        return

    raise AuthenticationError("Gecerli API anahtari gerekli.")


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings
        self.clients: dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if self.settings.enforce_https and request.url.scheme != "https":
            url = request.url.replace(scheme="https")
            return RedirectResponse(str(url), status_code=308)

        if self.settings.rate_limit_enabled and request.url.path.startswith(("/chat", "/api/")):
            limited_response = self._rate_limit(request)
            if limited_response is not None:
                return limited_response

        return await call_next(request)

    def _rate_limit(self, request: Request) -> JSONResponse | None:
        now = time.monotonic()
        window = self.settings.rate_limit_window_seconds
        limit = self.settings.rate_limit_requests
        client_id = self._client_id(request)
        hits = self.clients[client_id]

        while hits and now - hits[0] > window:
            hits.popleft()

        if len(hits) >= limit:
            retry_after = max(1, int(window - (now - hits[0])))
            return JSONResponse(
                status_code=429,
                content=error_payload(
                    code="rate_limit_exceeded",
                    message="Cok fazla istek gonderildi. Lutfen daha sonra tekrar deneyin.",
                    details={"limit": limit, "window_seconds": window},
                ),
                headers={"Retry-After": str(retry_after)},
            )

        hits.append(now)
        return None

    @staticmethod
    def _client_id(request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",", 1)[0].strip()
        if request.client:
            return request.client.host
        return "unknown"
