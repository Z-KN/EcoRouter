from pathlib import Path
import unittest


class TodoCoverageTests(unittest.TestCase):
    def test_deliberate_mvp_omissions_are_tracked(self) -> None:
        todo = Path("TODO.md").read_text(encoding="utf-8").lower()
        expected = (
            "text-only",
            "caller-supplied telemetry",
            "static capabilities",
            "approximate token",
            "ocr and image pii",
            "fastapi and websocket",
            "audio and general multimodal",
            "mlp or gbdt",
            "real-time pc dashboard",
            "production security",
            "evaluate and calibrate the regex privacy heuristic",
            "multilingual nlp models",
            "cirrascale streaming",
            "provider-side energy telemetry",
            "prefill and decode energy coefficients",
        )
        for item in expected:
            with self.subTest(item=item):
                self.assertIn(item, todo)


if __name__ == "__main__":
    unittest.main()
