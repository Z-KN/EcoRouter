"""Prompt feature extraction and privacy analysis."""

from __future__ import annotations

import math
import re
from typing import Protocol

from .models import Intent, PromptAnalysis

_SECRET_PATTERN = re.compile(
    r"\b(?:api[_ -]?key|access[_ -]?token|password|private[_ -]?key)\s*(?:is|=|:)\s*[^\s,;]+",
    re.IGNORECASE,
)
_ADDRESS_SUFFIX = (
    r"Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
    r"Court|Ct|Place|Pl|Way|Circle|Cir|Highway|Hwy|Parkway|Pkwy"
)
_HEURISTIC_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    ("payment_card", re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")),
    ("phone_number", re.compile(r"(?<!\w)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]\d{4}(?!\w)")),
    ("secret", _SECRET_PATTERN),
    # No NLP available on-device, so names/addresses are caught with a blunt
    # instrument: consecutive Title-Case words, and a house-number-plus-street-
    # suffix pattern. Both over-match (book titles, "United Kingdom", "Mona
    # Lisa" also trigger) -- deliberate, since over-blocking cloud is the safe
    # failure direction and under-blocking is the one that leaks PII.
    ("person", re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")),
    ("address", re.compile(rf"\b\d{{1,6}}\s+(?:[A-Z][a-zA-Z]*\s+){{1,4}}(?:{_ADDRESS_SUFFIX})\.?\b")),
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


class HeuristicPromptAnalyzer:
    """Dependency-free, regex-only prompt analyzer.

    Future work: an NLP-backed analyzer (e.g. Presidio, for NER-based
    PERSON/NRP detection and other entity types regex can't express) can be
    added later as another ``PromptAnalyzer`` implementation and swapped in
    without touching the router -- it only depends on this protocol.
    """

    def analyze(self, prompt: str) -> PromptAnalysis:
        categories = tuple(
            name for name, pattern in _HEURISTIC_PII_PATTERNS if pattern.search(prompt)
        )
        return _build_analysis(prompt, categories)


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
