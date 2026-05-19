from __future__ import annotations

from dataclasses import dataclass
import os


def _csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("ANAYASA_APP_NAME", "Anayasa AI")
    app_version: str = os.getenv("ANAYASA_APP_VERSION", "0.3.0")
    environment: str = os.getenv("ANAYASA_ENV", "development")
    log_level: str = os.getenv("ANAYASA_LOG_LEVEL", "INFO")

    cors_allowed_origins: tuple[str, ...] = tuple(
        _csv(
            os.getenv("ANAYASA_CORS_ORIGINS"),
            (
                "http://localhost:8000",
                "http://127.0.0.1:8000",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ),
        )
    )
    cors_allow_credentials: bool = _bool(os.getenv("ANAYASA_CORS_ALLOW_CREDENTIALS"), False)

    rate_limit_enabled: bool = _bool(os.getenv("ANAYASA_RATE_LIMIT_ENABLED"), True)
    rate_limit_requests: int = int(os.getenv("ANAYASA_RATE_LIMIT_REQUESTS", "20"))
    rate_limit_window_seconds: int = int(os.getenv("ANAYASA_RATE_LIMIT_WINDOW_SECONDS", "60"))

    query_min_length: int = int(os.getenv("ANAYASA_QUERY_MIN_LENGTH", "3"))
    query_max_length: int = int(os.getenv("ANAYASA_QUERY_MAX_LENGTH", "1500"))
    analysis_min_length: int = int(os.getenv("ANAYASA_ANALYSIS_MIN_LENGTH", "20"))
    analysis_max_length: int = int(os.getenv("ANAYASA_ANALYSIS_MAX_LENGTH", "5000"))

    api_key: str | None = os.getenv("ANAYASA_API_KEY")
    eager_load_rag: bool = _bool(os.getenv("ANAYASA_EAGER_LOAD_RAG"), True)
    enforce_https: bool = _bool(os.getenv("ANAYASA_ENFORCE_HTTPS"), False)


def get_settings() -> Settings:
    return Settings()
