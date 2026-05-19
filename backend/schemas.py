from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ConfidenceLevel = Literal["verified", "source_grounded", "needs_review"]
AnalysisMode = Literal["policy", "draft_text", "scenario"]
RiskLevel = Literal["low", "medium", "high", "critical"]
AnalysisStatus = Literal["no_clear_issue", "review_recommended", "high_risk", "needs_review"]


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


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    subject: str = Field(
        ...,
        min_length=3,
        max_length=160,
        description="Incelenecek politika, taslak veya olay basligi.",
        json_schema_extra={"example": "Belediye kamera izleme politikasi"},
    )
    description: str = Field(
        ...,
        min_length=20,
        max_length=5000,
        description="Anayasal risk acisindan incelenecek metin veya senaryo.",
        json_schema_extra={
            "example": "Belediye, tum meydanlarda yuz tanima destekli kamera sistemi kurmak istiyor."
        },
    )
    mode: AnalysisMode = Field(
        "policy",
        description="Inceleme turu: policy, draft_text veya scenario.",
    )

    @field_validator("subject", "description")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Metin bos olamaz.")
        return stripped


class AnalysisSignal(BaseModel):
    key: str = Field(..., description="Makine okunabilir risk sinyali anahtari.")
    label: str = Field(..., description="Kullaniciya gosterilecek risk sinyali basligi.")
    severity: RiskLevel = Field(..., description="Sinyalin agirlik seviyesi.")
    matched_terms: list[str] = Field(default_factory=list, description="Metinde eslesen terimler.")
    article_hints: list[str] = Field(
        default_factory=list,
        description="Sinyalle iliskili olabilecek anayasa maddesi ipuclari.",
    )


class AnalysisResponse(BaseModel):
    answer: str = Field(..., description="Anayasal inceleme raporu.")
    request_id: str = Field(..., description="Log takibi icin istek kimligi.")
    confidence: ConfidenceLevel = Field(
        "needs_review",
        description="Analizin kaynak ve dogrulama durumuna gore guven etiketi.",
    )
    status: AnalysisStatus = Field(..., description="Incelemenin ozet durum etiketi.")
    risk_level: RiskLevel = Field(..., description="Otomatik ve kaynakli bulgulara gore genel risk seviyesi.")
    signals: list[AnalysisSignal] = Field(
        default_factory=list,
        description="Metinden deterministik olarak cikarilan anayasal risk sinyalleri.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Analizi desteklemek icin getirilen kaynak parcalari.",
    )
    review_notes: list[str] = Field(
        default_factory=list,
        description="Kullanici ve reviewer icin kisa kontrol notlari.",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    rag_loaded: bool
