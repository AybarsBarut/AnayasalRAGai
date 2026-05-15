from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ConfidenceLevel = Literal["verified", "source_grounded", "needs_review"]


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


class Citation(BaseModel):
    article_id: str = Field(..., description="Dayanak anayasa maddesi veya gecici madde kimligi.")
    title: str = Field(..., description="Kaynak madde basligi.")
    excerpt: str = Field(..., description="RAG tarafindan kullanilan kaynak parcasi.")
    paragraph_index: int | None = Field(None, description="Kaynak parcanin madde icindeki sirasi.")
    source: str = Field("constitution.json", description="Kaynak veri dosyasi veya koleksiyon adi.")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="RAG sistemi tarafindan uretilen yanit.")
    request_id: str = Field(..., description="Log takibi icin istek kimligi.")
    confidence: ConfidenceLevel = Field(
        "needs_review",
        description="Yanitin kaynak ve alinti dogrulama durumuna gore guven etiketi.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Yaniti desteklemek icin getirilen kaynak parcalari.",
    )
    review_notes: list[str] = Field(
        default_factory=list,
        description="Kullanici ve reviewer icin kisa kontrol notlari.",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    rag_loaded: bool
