import json
import unittest

from ecorouter import (
    Device,
    DeviceConfig,
    DeviceTelemetry,
    EcoRouter,
    HeuristicPromptAnalyzer,
    NoRouteError,
    OptimizationProfile,
    RouteRequest,
    ValidationError,
    default_simulated_executors,
)
from ecorouter.scenarios import built_in_scenarios


def request_for(
    prompt: str,
    *,
    origin: Device = Device.PHONE,
    scenario: str = "healthy",
    profile: OptimizationProfile = OptimizationProfile.BALANCED,
) -> RouteRequest:
    return RouteRequest(prompt, origin, built_in_scenarios()[scenario], profile)


def heuristic_router(configs=None) -> EcoRouter:
    return EcoRouter(configs, analyzer=HeuristicPromptAnalyzer())


class EcoRouterTests(unittest.TestCase):
    def test_short_lookup_uses_efficient_phone_in_healthy_scenario(self) -> None:
        decision = heuristic_router().route(request_for("What's the weather tomorrow?"))

        self.assertEqual(decision.selected_device, Device.PHONE)
        self.assertFalse(decision.quality_degraded)

    def test_complex_reasoning_uses_cloud(self) -> None:
        prompt = (
            "First analyze this distributed architecture, then compare every trade-off step by step. "
            + "Include equations and detailed reasoning. " * 20
            + "Which design wins? What could fail?"
        )

        decision = heuristic_router().route(request_for(prompt))

        self.assertEqual(decision.selected_device, Device.CLOUD)

    def test_high_quality_profile_can_change_the_winner(self) -> None:
        balanced = heuristic_router().route(request_for("What's the weather tomorrow?"))
        high_quality = heuristic_router().route(
            request_for("What's the weather tomorrow?", profile=OptimizationProfile.HIGH_QUALITY)
        )

        self.assertEqual(balanced.selected_device, Device.PHONE)
        self.assertEqual(high_quality.selected_device, Device.CLOUD)

    def test_pii_blocks_cloud_and_diagnostics_do_not_contain_value(self) -> None:
        private_value = "alice@example.com"
        decision = heuristic_router().route(request_for(f"Summarize records for {private_value}"))
        cloud = next(item for item in decision.candidates if item.device == Device.CLOUD)

        self.assertFalse(cloud.eligible)
        self.assertIn("cloud blocked by privacy policy", cloud.exclusion_reasons)
        self.assertNotIn(private_value, json.dumps(decision.to_dict()))

    def test_sensitive_complex_request_stays_local_and_flags_degradation(self) -> None:
        prompt = (
            "Analyze records for alice@example.com. First compare every option step by step. "
            + "Detailed architecture equations and optimization requirements. " * 25
            + "What should change? What could fail?"
        )

        decision = heuristic_router().route(request_for(prompt))

        self.assertEqual(decision.selected_device, Device.PC)
        self.assertTrue(decision.quality_degraded)

    def test_low_battery_and_thermal_pressure_are_hard_gates(self) -> None:
        telemetry = built_in_scenarios()["phone-low-battery"]
        telemetry[Device.PC] = DeviceTelemetry(True, 18, 120, 0.025, 0.18, 0.95, 85, 0)

        decision = heuristic_router().route(RouteRequest("Hello", Device.PHONE, telemetry))

        self.assertEqual(decision.selected_device, Device.CLOUD)
        self.assertFalse(next(item for item in decision.candidates if item.device == Device.PHONE).eligible)
        self.assertFalse(next(item for item in decision.candidates if item.device == Device.PC).eligible)

    def test_no_route_when_all_destinations_are_blocked(self) -> None:
        telemetry = {
            Device.PHONE: DeviceTelemetry(False, 0, 0, 0, 0, 0, 50, 0),
            Device.PC: DeviceTelemetry(False, 0, 0, 0, 0, 0, 50, 0),
            Device.CLOUD: DeviceTelemetry(True, 30, 100, 0.1, 0.1, 0.1, None, 0.01),
        }

        with self.assertRaises(NoRouteError):
            heuristic_router().route(RouteRequest("Email alice@example.com", Device.PHONE, telemetry))

    def test_origin_sets_local_network_latency_to_zero(self) -> None:
        telemetry = {
            device: DeviceTelemetry(True, 100, 100, 0.01, 0.1, 0.1, None, 0)
            for device in Device
        }
        configs = {device: DeviceConfig(f"{device.value}-model", 0.60) for device in Device}
        configs[Device.CLOUD] = DeviceConfig("cloud-model", 0.50)

        decision = heuristic_router(configs).route(RouteRequest("Hello", Device.PC, telemetry))

        self.assertEqual(decision.selected_device, Device.PC)

    def test_exact_tie_uses_stable_device_order(self) -> None:
        telemetry = {
            device: DeviceTelemetry(True, 0, 100, 0.01, 0.1, 0.1, None, 0)
            for device in Device
        }
        configs = {device: DeviceConfig(f"{device.value}-model", 0.60) for device in Device}

        decision = heuristic_router(configs).route(
            RouteRequest("Contact alice@example.com", Device.PHONE, telemetry)
        )

        self.assertEqual(decision.selected_device, Device.PHONE)

    def test_simulated_executor_dispatches_without_echoing_prompt(self) -> None:
        private_value = "alice@example.com"
        result = heuristic_router().run(
            request_for(f"Summarize {private_value}"), default_simulated_executors()
        )

        self.assertIn(result.decision.model_id, result.response)
        self.assertNotIn(private_value, result.response)

    def test_request_requires_all_three_telemetry_entries(self) -> None:
        with self.assertRaises(ValidationError):
            RouteRequest("Hello", Device.PHONE, {Device.PHONE: built_in_scenarios()["healthy"][Device.PHONE]})

    def test_telemetry_rejects_string_numbers_and_non_boolean_availability(self) -> None:
        with self.assertRaises(ValidationError):
            DeviceTelemetry(True, "10", 100, 0.01, 0.1, 0.1, 80, 0)
        with self.assertRaises(ValidationError):
            DeviceTelemetry("true", 10, 100, 0.01, 0.1, 0.1, 80, 0)


if __name__ == "__main__":
    unittest.main()
