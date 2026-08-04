"""Prompt feature extraction and privacy analysis."""

from __future__ import annotations

from functools import lru_cache
import math
import re
from typing import Any, Protocol

from .models import (
    Intent,
    PrivacyAnalysisError,
    PrivacyInitializationError,
    PromptAnalysis,
    ValidationError,
)

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
except ImportError:  # Preserves an actionable fail-closed error for incomplete environments.
    AnalyzerEngine = None  # type: ignore[assignment,misc]
    NlpEngineProvider = None  # type: ignore[assignment,misc]


PRESIDIO_LANGUAGE = "en"
PRESIDIO_MODEL = "en_core_web_sm"
PRESIDIO_SCORE_THRESHOLD = 0.50

# The policy deliberately excludes generic LOCATION, DATE_TIME, and URL entities.
_PRESIDIO_ENTITY_POLICY: tuple[tuple[str, str], ...] = (
    ("PERSON", "person"),
    ("NRP", "nrp"),
    ("EMAIL_ADDRESS", "email"),
    ("PHONE_NUMBER", "phone_number"),
    ("CREDIT_CARD", "payment_card"),
    ("IBAN_CODE", "iban_code"),
    ("US_BANK_NUMBER", "us_bank_number"),
    ("CRYPTO", "crypto"),
    ("US_SSN", "ssn"),
    ("US_PASSPORT", "us_passport"),
    ("US_DRIVER_LICENSE", "us_driver_license"),
    ("US_ITIN", "us_itin"),
    ("MEDICAL_LICENSE", "medical_license"),
    ("IP_ADDRESS", "ip_address"),
    ("MAC_ADDRESS", "mac_address"),
)
PRESIDIO_SENSITIVE_ENTITIES = tuple(item[0] for item in _PRESIDIO_ENTITY_POLICY)
_PRESIDIO_CATEGORY_NAMES = dict(_PRESIDIO_ENTITY_POLICY)

_SECRET_PATTERN = re.compile(
    r"\b(?:api[_ -]?key|access[_ -]?token|password|private[_ -]?key)\s*(?:is|=|:)\s*[^\s,;]+",
    re.IGNORECASE,
)
_HEURISTIC_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    ("payment_card", re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")),
    ("phone_number", re.compile(r"(?<!\w)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]\d{4}(?!\w)")),
    ("secret", _SECRET_PATTERN),
)

_INTENT_PATTERNS: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (Intent.CODING, re.compile(r"\b(?:code|debug|function|class|python|javascript|sql|algorithm|compile)\b", re.I)),
    (Intent.SUMMARIZATION, re.compile(r"\b(?:summari[sz]e|summary|condense|key points|tl;dr)\b", re.I)),
    (Intent.CREATIVE, re.compile(r"\b(?:write a story|poem|brainstorm|creative|slogan|screenplay)\b", re.I)),
    (Intent.REASONING, re.compile(r"\b(?:analy[sz]e|compare|derive|prove|reason|evaluate|trade-?offs?|step by step)\b", re.I)),
    (Intent.LOOKUP, re.compile(r"\b(?:what|when|where|who|weather|define|lookup|find)\b", re.I)),
)

_MULTI_STEP = re.compile(
    r"\b(?:first|second|third|then|next|finally|step\s+\d+|and then|checklist|action plan)\b",
    re.IGNORECASE,
)
_COMPLEX_REASONING = re.compile(
    r"\b(?:analy[sz]e|compare|derive|prove|reason|algorithm|debug|optimi[sz]e|architecture|equation|calculate)\b",
    re.IGNORECASE,
)

_OUTPUT_TOKENS = {
    Intent.LOOKUP: 96,
    Intent.SUMMARIZATION: 256,
    Intent.CODING: 512,
    Intent.REASONING: 384,
    Intent.CREATIVE: 384,
    Intent.GENERAL: 192,
}


class PromptAnalyzer(Protocol):
    """Contract consumed by the routing policy."""

    def analyze(self, prompt: str) -> PromptAnalysis:
        """Return privacy-safe prompt features without retaining entity values."""


class PresidioEngine(Protocol):
    """Minimal AnalyzerEngine surface used for dependency injection in tests."""

    def analyze(
        self,
        *,
        text: str,
        language: str,
        entities: list[str],
        score_threshold: float,
    ) -> list[Any]: ...


class HeuristicPromptAnalyzer:
    """Legacy dependency-free analyzer for explicit tests and compatibility."""

    def analyze(self, prompt: str) -> PromptAnalysis:
        categories = tuple(
            name for name, pattern in _HEURISTIC_PII_PATTERNS if pattern.search(prompt)
        )
        return _build_analysis(prompt, categories)


class PresidioPromptAnalyzer:
    """Detect policy-selected PII with a local Presidio AnalyzerEngine."""

    def __init__(
        self,
        engine: PresidioEngine | None = None,
        *,
        score_threshold: float = PRESIDIO_SCORE_THRESHOLD,
    ) -> None:
        if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
            raise ValidationError("score_threshold must be a number")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValidationError("score_threshold must be between 0.0 and 1.0")
        self.score_threshold = float(score_threshold)
        self.engine = engine if engine is not None else _cached_presidio_engine()

    def analyze(self, prompt: str) -> PromptAnalysis:
        try:
            results = self.engine.analyze(
                text=prompt,
                language=PRESIDIO_LANGUAGE,
                entities=list(PRESIDIO_SENSITIVE_ENTITIES),
                score_threshold=self.score_threshold,
            )
        except Exception:
            raise PrivacyAnalysisError(
                "Presidio could not analyze the prompt; routing stopped to preserve privacy."
            ) from None

        detected = {
            result.entity_type
            for result in results
            if result.entity_type in _PRESIDIO_CATEGORY_NAMES
            and result.score >= self.score_threshold
        }
        categories = [
            category
            for entity, category in _PRESIDIO_ENTITY_POLICY
            if entity in detected
        ]
        if _SECRET_PATTERN.search(prompt):
            categories.append("secret")
        return _build_analysis(prompt, tuple(categories))


@lru_cache(maxsize=1)
def _cached_presidio_engine() -> PresidioEngine:
    if AnalyzerEngine is None or NlpEngineProvider is None:
        raise PrivacyInitializationError(
            "Presidio is unavailable; install the project in its virtual environment with "
            "'python -m pip install -e .'."
        )
    try:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": PRESIDIO_LANGUAGE, "model_name": PRESIDIO_MODEL}
                ],
            }
        )
        nlp_engine = provider.create_engine()
        engine = AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=[PRESIDIO_LANGUAGE]
        )
        _use_offline_email_validation(engine)
        return engine
    except Exception as error:
        raise PrivacyInitializationError(
            "Presidio could not initialize the en_core_web_sm model; reinstall project "
            "dependencies before routing prompts."
        ) from error


def _use_offline_email_validation(engine: Any) -> None:
    """Use tldextract's bundled suffix snapshot without network refreshes."""
    from presidio_analyzer.predefined_recognizers import EmailRecognizer
    from tldextract import TLDExtract

    class OfflineEmailRecognizer(EmailRecognizer):
        _extractor = TLDExtract(cache_dir=None, suffix_list_urls=())

        def validate_result(self, pattern_text: str) -> bool:
            return self._extractor(pattern_text).fqdn != ""

    engine.registry.remove_recognizer("EmailRecognizer", language=PRESIDIO_LANGUAGE)
    engine.registry.add_recognizer(OfflineEmailRecognizer())


def _build_analysis(prompt: str, categories: tuple[str, ...]) -> PromptAnalysis:
    intent = _intent(prompt)
    input_tokens = max(1, math.ceil(len(prompt) / 4))

    complexity = 0.15
    if input_tokens > 80:
        complexity += 0.20
    if input_tokens > 200:
        complexity += 0.20
    if _MULTI_STEP.search(prompt):
        complexity += 0.20
    if _COMPLEX_REASONING.search(prompt):
        complexity += 0.20
    if prompt.count("?") > 1:
        complexity += 0.10
    complexity = min(1.0, complexity)

    return PromptAnalysis(
        intent=intent,
        complexity=complexity,
        sensitive=bool(categories),
        pii_categories=categories,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=_OUTPUT_TOKENS[intent],
        required_quality=min(0.95, 0.45 + 0.50 * complexity),
    )


def _intent(prompt: str) -> Intent:
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(prompt):
            return intent
    return Intent.GENERAL
