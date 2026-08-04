from dataclasses import dataclass
import sys
import traceback
import unittest
from unittest.mock import patch

from ecorouter import (
    HeuristicPromptAnalyzer,
    Intent,
    PresidioPromptAnalyzer,
    PrivacyAnalysisError,
    PrivacyInitializationError,
    ValidationError,
)
from ecorouter.analyzer import PRESIDIO_SENSITIVE_ENTITIES, _cached_presidio_engine


@dataclass
class FakeRecognizerResult:
    entity_type: str
    score: float


class FakePresidioEngine:
    def __init__(self, results=None, error=None) -> None:
        self.results = list(results or [])
        self.error = error
        self.calls = []

    def analyze(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results


class HeuristicPromptAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = HeuristicPromptAnalyzer()

    def test_detects_pii_categories_without_returning_values(self) -> None:
        prompt = "Email alice@example.com or call 217-555-0199; password=top-secret"

        analysis = self.analyzer.analyze(prompt)

        self.assertTrue(analysis.sensitive)
        self.assertEqual(analysis.pii_categories, ("email", "phone_number", "secret"))
        self.assertNotIn("alice@example.com", repr(analysis))
        self.assertNotIn("top-secret", repr(analysis))

    def test_classifies_intent_using_documented_precedence(self) -> None:
        self.assertEqual(self.analyzer.analyze("Debug this Python function").intent, Intent.CODING)
        self.assertEqual(self.analyzer.analyze("Summarize these notes").intent, Intent.SUMMARIZATION)
        self.assertEqual(self.analyzer.analyze("What's the weather?").intent, Intent.LOOKUP)

    def test_multi_step_reasoning_is_more_complex_than_lookup(self) -> None:
        lookup = self.analyzer.analyze("What's the weather tomorrow?")
        reasoning = self.analyzer.analyze(
            "First analyze the architecture, then compare the trade-offs step by step. "
            "What is fastest? What is most efficient?"
        )

        self.assertGreater(reasoning.complexity, lookup.complexity)
        self.assertGreater(reasoning.required_quality, lookup.required_quality)
        self.assertGreater(reasoning.estimated_output_tokens, lookup.estimated_output_tokens)


class PresidioPromptAnalyzerTests(unittest.TestCase):
    def test_filters_threshold_maps_categories_and_deduplicates(self) -> None:
        private_value = "John Smith"
        engine = FakePresidioEngine(
            [
                FakeRecognizerResult("PERSON", 0.91),
                FakeRecognizerResult("PERSON", 0.85),
                FakeRecognizerResult("EMAIL_ADDRESS", 0.80),
                FakeRecognizerResult("CREDIT_CARD", 0.49),
                FakeRecognizerResult("LOCATION", 0.99),
            ]
        )
        analyzer = PresidioPromptAnalyzer(engine=engine)

        analysis = analyzer.analyze(
            f"{private_value} has email alice@example.com and password=top-secret"
        )

        self.assertEqual(analysis.pii_categories, ("person", "email", "secret"))
        self.assertTrue(analysis.sensitive)
        self.assertNotIn(private_value, repr(analysis))
        self.assertEqual(engine.calls[0]["score_threshold"], 0.50)
        self.assertEqual(engine.calls[0]["entities"], list(PRESIDIO_SENSITIVE_ENTITIES))
        self.assertNotIn("LOCATION", engine.calls[0]["entities"])
        self.assertNotIn("DATE_TIME", engine.calls[0]["entities"])
        self.assertNotIn("URL", engine.calls[0]["entities"])

    def test_secret_regex_supplements_empty_presidio_results(self) -> None:
        analysis = PresidioPromptAnalyzer(engine=FakePresidioEngine()).analyze(
            "Use api_key:abc123"
        )

        self.assertEqual(analysis.pii_categories, ("secret",))

    def test_runtime_failure_stops_privacy_analysis_without_prompt_text(self) -> None:
        private_value = "John Smith"
        analyzer = PresidioPromptAnalyzer(
            engine=FakePresidioEngine(error=RuntimeError(private_value))
        )

        try:
            analyzer.analyze(f"My name is {private_value}")
        except PrivacyAnalysisError as error:
            caught_error = error
            rendered_error = "".join(traceback.format_exception(*sys.exc_info()))
        else:
            self.fail("Presidio runtime failure did not stop routing")

        self.assertNotIn(private_value, str(caught_error))
        self.assertNotIn(private_value, rendered_error)

    def test_missing_presidio_fails_closed_during_initialization(self) -> None:
        _cached_presidio_engine.cache_clear()
        with (
            patch("ecorouter.analyzer.AnalyzerEngine", None),
            patch("ecorouter.analyzer.NlpEngineProvider", None),
        ):
            with self.assertRaises(PrivacyInitializationError):
                PresidioPromptAnalyzer()
        _cached_presidio_engine.cache_clear()

    def test_score_threshold_is_validated(self) -> None:
        with self.assertRaises(ValidationError):
            PresidioPromptAnalyzer(engine=FakePresidioEngine(), score_threshold=1.1)


if __name__ == "__main__":
    unittest.main()
