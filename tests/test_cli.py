import contextlib
import io
import json
import unittest
from unittest.mock import patch

from ecorouter.cli import main
from ecorouter.models import PrivacyInitializationError


class CliTests(unittest.TestCase):
    def test_route_emits_machine_readable_json(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(
                ["route", "--origin", "phone", "--prompt", "What's the weather?", "--json"]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["selected_device"], "phone")
        self.assertNotIn("prompt", payload)

    def test_run_prints_simulated_response(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["run", "--origin", "pc", "--prompt", "Summarize this note"])

        self.assertEqual(code, 0)
        self.assertIn("Response: Simulated", stdout.getvalue())

    def test_empty_prompt_returns_input_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["route", "--origin", "phone", "--prompt", ""])

        self.assertEqual(code, 2)
        self.assertIn("prompt must not be empty", stderr.getvalue())

    def test_all_built_in_scenarios_execute(self) -> None:
        for scenario in ("healthy", "phone-low-battery", "pc-congested", "cloud-offline"):
            with self.subTest(scenario=scenario):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = main(
                        [
                            "route",
                            "--origin",
                            "phone",
                            "--prompt",
                            "Hello",
                            "--scenario",
                            scenario,
                        ]
                    )
                self.assertEqual(code, 0)
                self.assertIn("Selected:", stdout.getvalue())

    def test_privacy_initialization_failure_returns_exit_code_five(self) -> None:
        stderr = io.StringIO()
        with (
            patch(
                "ecorouter.cli.EcoRouter",
                side_effect=PrivacyInitializationError("privacy runtime unavailable"),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(["route", "--origin", "phone", "--prompt", "Hello"])

        self.assertEqual(code, 5)
        self.assertEqual(
            stderr.getvalue().strip(),
            "privacy error: privacy runtime unavailable",
        )


if __name__ == "__main__":
    unittest.main()
