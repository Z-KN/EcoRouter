import unittest

from peqrouter import HeuristicPromptAnalyzer, Intent


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

    def test_detects_person_and_address(self) -> None:
        analysis = self.analyzer.analyze(
            "Patient Maria Garcia, date of birth 1984-03-12, needs a discharge summary."
        )

        self.assertTrue(analysis.sensitive)
        self.assertEqual(analysis.pii_categories, ("person",))

        analysis = self.analyzer.analyze(
            "Employee Robert Chen, who lives at 42 Oak Street, needs a performance review."
        )

        self.assertTrue(analysis.sensitive)
        self.assertEqual(analysis.pii_categories, ("person", "address"))


if __name__ == "__main__":
    unittest.main()
