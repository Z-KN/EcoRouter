import json
import unittest
from unittest.mock import patch

from ecorouter import Device, EcoRouter, RouteRequest
from ecorouter.scenarios import built_in_scenarios


def presidio_request(prompt: str) -> RouteRequest:
    return RouteRequest(
        prompt=prompt,
        origin=Device.PHONE,
        telemetry=built_in_scenarios()["healthy"],
    )


class PresidioIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.router = EcoRouter()

    def test_full_person_name_is_sensitive_and_blocks_cloud(self) -> None:
        private_value = "John Smith"

        decision = self.router.route(
            presidio_request(f"Summarize the profile for {private_value}")
        )
        cloud = next(item for item in decision.candidates if item.device == Device.CLOUD)

        self.assertIn("person", decision.analysis.pii_categories)
        self.assertTrue(decision.analysis.sensitive)
        self.assertFalse(cloud.eligible)
        self.assertNotIn(private_value, json.dumps(decision.to_dict()))

    def test_structured_identifiers_remain_sensitive(self) -> None:
        with patch(
            "requests.sessions.Session.get",
            side_effect=AssertionError("privacy analysis attempted network access"),
        ):
            decision = self.router.route(
                presidio_request(
                    "Contact alice@example.com. My phone number is +1 415-555-2671. "
                    "My SSN is 123-45-6790 and card 4111 1111 1111 1111."
                )
            )

        self.assertTrue(decision.analysis.sensitive)
        self.assertTrue(
            {"email", "phone_number", "ssn", "payment_card"}.issubset(
                decision.analysis.pii_categories
            )
        )

    def test_generic_location_date_and_url_are_not_sensitive(self) -> None:
        decision = self.router.route(
            presidio_request(
                "What's the weather in Seattle tomorrow? Use https://weather.example.com"
            )
        )

        self.assertFalse(decision.analysis.sensitive)
        self.assertEqual(decision.analysis.pii_categories, ())

    def test_complex_named_person_prompt_stays_local_with_degraded_quality(self) -> None:
        prompt = (
            "Analyze records for John Smith. First compare every option step by step. "
            + "Detailed architecture equations and optimization requirements. " * 25
            + "What should change? What could fail?"
        )

        decision = self.router.route(presidio_request(prompt))

        self.assertEqual(decision.selected_device, Device.PC)
        self.assertTrue(decision.quality_degraded)


if __name__ == "__main__":
    unittest.main()
