from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        ...,
        min_length=3,
        max_length=1500,
        description="Anayasa hakkinda yanitlanacak kullanici sorusu.",
        json_schema_extra={"example": "Anayasa'nin 1. maddesi nedir?"},
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Soru bos olamaz.")
        return stripped


class ChatResponse(BaseModel):
    answer: str = Field(..., description="RAG sistemi tarafindan uretilen yanit.")
    request_id: str = Field(..., description="Log takibi icin istek kimligi.")


class HealthResponse(BaseModel):
    status: str
    version: str
    rag_loaded: bool
