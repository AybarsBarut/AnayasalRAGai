from __future__ import annotations

import re
from typing import Any, Iterable


RISK_LEVEL_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}

CONSTITUTIONAL_RISK_TAXONOMY: tuple[dict[str, Any], ...] = (
    {
        "key": "fundamental_right_limitation",
        "label": "Temel hak ve hürriyet sınırlaması",
        "severity": "high",
        "article_hints": ("13", "15"),
        "terms": (
            "sınırlama",
            "yasak",
            "erişim engeli",
            "askıya alma",
            "durdurma",
            "müdahale",
            "kısıtlama",
        ),
    },
    {
        "key": "equality_and_discrimination",
        "label": "Eşitlik ve ayrımcılık riski",
        "severity": "high",
        "article_hints": ("10",),
        "terms": (
            "ayrımcılık",
            "eşitlik",
            "cinsiyet",
            "ırk",
            "dil",
            "din",
            "mezhep",
            "siyasi düşünce",
            "engellilik",
        ),
    },
    {
        "key": "private_life_and_personal_data",
        "label": "Özel hayat ve kişisel veri riski",
        "severity": "high",
        "article_hints": ("20",),
        "terms": (
            "özel hayat",
            "kişisel veri",
            "veri işleme",
            "gözetim",
            "kamera",
            "biyometrik",
            "mahremiyet",
        ),
    },
    {
        "key": "expression_and_media",
        "label": "İfade, basın ve haberleşme özgürlüğü riski",
        "severity": "high",
        "article_hints": ("22", "26", "28"),
        "terms": (
            "ifade özgürlüğü",
            "basın",
            "sansür",
            "yayın yasağı",
            "haberleşme",
            "internet",
            "sosyal medya",
        ),
    },
    {
        "key": "assembly_and_association",
        "label": "Toplantı, gösteri ve örgütlenme riski",
        "severity": "medium",
        "article_hints": ("33", "34"),
        "terms": (
            "toplantı",
            "gösteri",
            "dernek",
            "örgütlenme",
            "sendika",
            "yürüyüş",
        ),
    },
    {
        "key": "property_and_expropriation",
        "label": "Mülkiyet ve kamulaştırma riski",
        "severity": "medium",
        "article_hints": ("35", "46"),
        "terms": (
            "mülkiyet",
            "kamulaştırma",
            "el koyma",
            "taşınmaz",
            "bedel",
            "malvarlığı",
        ),
    },
    {
        "key": "judicial_protection",
        "label": "Hak arama ve yargısal denetim riski",
        "severity": "high",
        "article_hints": ("36", "40", "125"),
        "terms": (
            "itiraz yolu",
            "dava açma",
            "yargı yolu",
            "mahkeme",
            "savunma hakkı",
            "başvuru hakkı",
        ),
    },
    {
        "key": "criminal_and_sanction",
        "label": "Ceza, yaptırım ve ölçülülük riski",
        "severity": "medium",
        "article_hints": ("38",),
        "terms": (
            "ceza",
            "idari para cezası",
            "yaptırım",
            "suç",
            "tutuklama",
            "gözaltı",
            "hapis",
        ),
    },
    {
        "key": "administrative_authority",
        "label": "İdarenin yetkisi ve kanunilik riski",
        "severity": "medium",
        "article_hints": ("6", "7", "8", "123", "124"),
        "terms": (
            "idare",
            "kurum",
            "yetki",
            "yönetmelik",
            "genelge",
            "karar",
            "resen",
        ),
    },
)


def build_analysis_query(subject: str, description: str, mode: str) -> str:
    mode_label = {
        "policy": "politika veya işlem",
        "draft_text": "taslak düzenleme",
        "scenario": "somut olay",
    }.get(mode, "metin")
    return (
        f"Konu: {subject}\n"
        f"İnceleme türü: {mode_label}\n\n"
        f"{description}\n\n"
        "Bu metni yalnızca Türkiye Cumhuriyeti Anayasası bağlamında; temel haklar, "
        "yetki, usul, ölçülülük ve kaynak dayanağı açısından incele."
    )


def detect_risk_signals(text: str) -> list[dict[str, Any]]:
    normalized = _normalize(text)
    signals: list[dict[str, Any]] = []

    for item in CONSTITUTIONAL_RISK_TAXONOMY:
        matched_terms = _matched_terms(normalized, item["terms"])
        if not matched_terms:
            continue
        signals.append(
            {
                "key": item["key"],
                "label": item["label"],
                "severity": item["severity"],
                "matched_terms": matched_terms,
                "article_hints": list(item["article_hints"]),
            }
        )

    return sorted(
        signals,
        key=lambda signal: RISK_LEVEL_ORDER.get(signal["severity"], 0),
        reverse=True,
    )


def risk_level_from_signals(
    signals: Iterable[dict[str, Any]],
    confidence: str,
    citations: Iterable[dict[str, Any]],
) -> str:
    if confidence == "needs_review" and not list(citations):
        return "high"

    highest = "low"
    for signal in signals:
        severity = str(signal.get("severity", "low"))
        if RISK_LEVEL_ORDER.get(severity, 0) > RISK_LEVEL_ORDER[highest]:
            highest = severity
    return highest


def analysis_status(risk_level: str, confidence: str, citations: Iterable[dict[str, Any]]) -> str:
    if confidence == "needs_review" or not list(citations):
        return "needs_review"
    if risk_level in {"critical", "high"}:
        return "high_risk"
    if risk_level == "medium":
        return "review_recommended"
    return "no_clear_issue"


def signal_summary(signals: Iterable[dict[str, Any]]) -> str:
    lines = []
    for signal in signals:
        terms = ", ".join(signal.get("matched_terms", [])[:5])
        hints = ", ".join(f"Madde {hint}" for hint in signal.get("article_hints", []))
        lines.append(f"- {signal.get('label')} ({signal.get('severity')}): {terms}; olası dayanaklar: {hints}")
    return "\n".join(lines) if lines else "- Belirgin otomatik risk sinyali bulunmadı."


def _matched_terms(normalized_text: str, terms: Iterable[str]) -> list[str]:
    matched = []
    for term in terms:
        normalized_term = _normalize(term)
        pattern = rf"(?<!\w){re.escape(normalized_term)}(?!\w)"
        if re.search(pattern, normalized_text):
            matched.append(term)
    return matched


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()
