import json
import unittest
from dataclasses import replace

from peqrouter import (
    Device,
    DeviceConfig,
    DeviceTelemetry,
    PEQRouter,
    ExecutionObservation,
    HeuristicPromptAnalyzer,
    NoRouteError,
    OptimizationProfile,
    RouteRequest,
    ValidationError,
    default_simulated_executors,
)
from peqrouter.estimator import EstimatorUnavailableError, PromptEstimate
from peqrouter.scenarios import built_in_scenarios


def request_for(
    prompt: str,
    *,
    origin: Device = Device.PHONE,
    scenario: str = "healthy",
    profile: OptimizationProfile = OptimizationProfile.BALANCED,
) -> RouteRequest:
    return RouteRequest(prompt, origin, built_in_scenarios()[scenario], profile)


def heuristic_router(configs=None, estimator=None) -> PEQRouter:
    return PEQRouter(configs, analyzer=HeuristicPromptAnalyzer(), estimator=estimator)


def _untrusted_estimate() -> PromptEstimate:
    return PromptEstimate(
        p_pass={},
        length_p50={},
        length_p90={},
        confidence="low",
        mean_distance=1.0,
        neighbours=(),
    )


class PEQRouterTests(unittest.TestCase):
    def test_observed_executor_adds_rounded_live_metrics_and_energy_estimate(self) -> None:
        class ObservedCloudExecutor:
            def execute(self, prompt, decision):
                raise AssertionError("legacy execute must not be called")

            def execute_observed(self, prompt, decision):
                return ExecutionObservation(
                    response="observed response",
                    api_turnaround_latency_ms=125.1236,
                    model_id=decision.model_id,
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                )

        executors = default_simulated_executors()
        executors[Device.CLOUD] = ObservedCloudExecutor()
        result = heuristic_router().run(
            request_for(
                "What model are you?",
                origin=Device.PC,
                profile=OptimizationProfile.HIGH_QUALITY,
            ),
            executors,
        )

        self.assertEqual(result.response, "observed response")
        self.assertIsNotNone(result.metrics)
        self.assertAlmostEqual(result.metrics.estimated_energy_joules, 67.425832, places=5)
        self.assertIsNone(result.metrics.measured_energy_joules)
        self.assertAlmostEqual(result.metrics.energy_joules_per_token, 2.247528, places=5)
        self.assertEqual(result.metrics.confidence, "uncalibrated")
        self.assertIn("cloud", result.metrics.energy_scope)
        self.assertEqual(result.to_dict()["metrics"]["api_turnaround_latency_ms"], 125.124)
        self.assertEqual(result.to_dict()["metrics"]["estimated_energy_joules"], 67.425832)

    def test_uncalibrated_pc_estimate_is_labeled_pc_not_cloud(self) -> None:
        class ObservedPcExecutor:
            def execute(self, prompt, decision):
                raise AssertionError("legacy execute must not be called")

            def execute_observed(self, prompt, decision):
                return ExecutionObservation(
                    response="pc response",
                    api_turnaround_latency_ms=93.0,
                    model_id=decision.model_id,
                    prompt_tokens=30,
                    completion_tokens=22,
                    total_tokens=52,
                )

        # Phone/cloud made unavailable so PC wins by elimination -- this test
        # is about the uncalibrated-energy-fallback labeling, not about which
        # device the latency/energy/quality tradeoff would otherwise favour.
        telemetry = dict(built_in_scenarios()["healthy"])
        telemetry[Device.PHONE] = replace(telemetry[Device.PHONE], available=False)
        telemetry[Device.CLOUD] = replace(telemetry[Device.CLOUD], available=False)

        executors = default_simulated_executors()
        executors[Device.PC] = ObservedPcExecutor()
        result = heuristic_router().run(
            RouteRequest("What model are you?", Device.PC, telemetry, OptimizationProfile.LOW_LATENCY),
            executors,
        )

        self.assertEqual(result.decision.selected_device, Device.PC)
        self.assertEqual(result.metrics.confidence, "uncalibrated")
        self.assertIn("PC (X-Elite NPU)", result.metrics.energy_scope)
        self.assertNotIn("cloud", result.metrics.energy_scope)

    def test_measured_phone_energy_is_passed_through_with_full_stats(self) -> None:
        class ObservedPhoneExecutor:
            def execute(self, prompt, decision):
                raise AssertionError("legacy execute must not be called")

            def execute_observed(self, prompt, decision):
                return ExecutionObservation(
                    response="phone response",
                    api_turnaround_latency_ms=410.0,
                    model_id=decision.model_id,
                    prompt_tokens=12,
                    completion_tokens=40,
                    total_tokens=52,
                    ttft_ms=88.5,
                    prefill_speed_tokens_per_second=140.2,
                    decode_speed_tokens_per_second=18.6,
                    measured_energy_joules=0.512,
                    tokens_per_joule=78.1,
                    compute_unit="npu",
                )

        # Flat capability scores isolate the energy-telemetry assertions below
        # from the quality penalty -- this test is about the passthrough, not
        # about which device the quality/latency/energy tradeoff favours.
        configs = {device: DeviceConfig(f"{device.value}-model", 0.80) for device in Device}
        executors = default_simulated_executors()
        executors[Device.PHONE] = ObservedPhoneExecutor()
        result = heuristic_router(configs).run(
            request_for("What model are you?", profile=OptimizationProfile.LOW_ENERGY),
            executors,
        )

        self.assertEqual(result.decision.selected_device, Device.PHONE)
        self.assertEqual(result.metrics.measured_energy_joules, 0.512)
        self.assertEqual(result.metrics.confidence, "measured")
        self.assertIn("measured whole-device battery discharge", result.metrics.energy_scope)
        self.assertEqual(result.metrics.ttft_ms, 88.5)
        self.assertEqual(result.metrics.prefill_speed_tokens_per_second, 140.2)
        self.assertEqual(result.metrics.decode_speed_tokens_per_second, 18.6)
        self.assertEqual(result.metrics.tokens_per_joule, 78.1)
        self.assertEqual(result.metrics.compute_unit, "npu")
        # Estimated (telemetry-table) energy is still computed alongside the
        # measured figure -- useful as a comparison point, not a replacement.
        self.assertIsNotNone(result.metrics.estimated_energy_joules)

    def test_measured_pc_energy_is_passed_through_with_full_stats(self) -> None:
        class ObservedPcExecutor:
            def execute(self, prompt, decision):
                raise AssertionError("legacy execute must not be called")

            def execute_observed(self, prompt, decision):
                return ExecutionObservation(
                    response="pc response",
                    api_turnaround_latency_ms=1500.0,
                    model_id=decision.model_id,
                    prompt_tokens=26,
                    completion_tokens=153,
                    total_tokens=179,
                    ttft_ms=45.3,
                    prefill_speed_tokens_per_second=900.0,
                    decode_speed_tokens_per_second=21.0,
                    measured_energy_joules=9.1467,
                    tokens_per_joule=2.4,
                    compute_unit="npu",
                    backend="geniex",
                )

        # Phone/cloud made unavailable so PC wins by elimination -- this test
        # is about the measured-energy passthrough, not about which device
        # the latency/energy/quality tradeoff would otherwise favour.
        telemetry = dict(built_in_scenarios()["healthy"])
        telemetry[Device.PHONE] = replace(telemetry[Device.PHONE], available=False)
        telemetry[Device.CLOUD] = replace(telemetry[Device.CLOUD], available=False)

        executors = default_simulated_executors()
        executors[Device.PC] = ObservedPcExecutor()
        result = heuristic_router().run(
            RouteRequest("What model are you?", Device.PC, telemetry, OptimizationProfile.LOW_LATENCY),
            executors,
        )

        self.assertEqual(result.decision.selected_device, Device.PC)
        self.assertAlmostEqual(result.metrics.measured_energy_joules, 9.1467)
        self.assertEqual(result.metrics.confidence, "measured")
        self.assertIn("measured whole-laptop battery discharge", result.metrics.energy_scope)
        self.assertNotIn("phone is unplugged", result.metrics.energy_scope)
        self.assertEqual(result.metrics.ttft_ms, 45.3)
        self.assertEqual(result.metrics.prefill_speed_tokens_per_second, 900.0)
        self.assertEqual(result.metrics.decode_speed_tokens_per_second, 21.0)
        self.assertEqual(result.metrics.tokens_per_joule, 2.4)
        self.assertEqual(result.metrics.compute_unit, "npu")
        self.assertEqual(result.metrics.backend, "geniex")
        self.assertIsNotNone(result.metrics.estimated_energy_joules)

    def test_phone_energy_falls_back_to_uncalibrated_when_unmeasured(self) -> None:
        """A phone observation without measured energy (e.g. non-NPU compute unit,
        or the model isn't in the power table) must not be mislabeled as measured."""

        class ObservedPhoneExecutor:
            def execute(self, prompt, decision):
                raise AssertionError("legacy execute must not be called")

            def execute_observed(self, prompt, decision):
                return ExecutionObservation(
                    response="phone response",
                    api_turnaround_latency_ms=410.0,
                    model_id=decision.model_id,
                    prompt_tokens=12,
                    completion_tokens=40,
                    total_tokens=52,
                    compute_unit="cpu",
                )

        configs = {device: DeviceConfig(f"{device.value}-model", 0.80) for device in Device}
        executors = default_simulated_executors()
        executors[Device.PHONE] = ObservedPhoneExecutor()
        result = heuristic_router(configs).run(
            request_for("What model are you?", profile=OptimizationProfile.LOW_ENERGY),
            executors,
        )

        self.assertIsNone(result.metrics.measured_energy_joules)
        self.assertEqual(result.metrics.confidence, "uncalibrated")
        self.assertIn("phone", result.metrics.energy_scope)

    def test_short_lookup_in_balanced_profile(self) -> None:
        # Real calibrated constants: PC is both slower and far more
        # energy-hungry per token than phone on this hardware pairing (see
        # scenarios.py), so phone wins a short lookup under balanced despite
        # its lower static capability_score.
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

    def test_low_battery_and_thermal_pressure_no_longer_affect_routing(self) -> None:
        # Battery and thermal pressure are no longer hard gates or scored
        # penalties -- only latency, energy, and quality drive routing now.
        telemetry = built_in_scenarios()["phone-low-battery"]
        telemetry[Device.PC] = DeviceTelemetry(True, 18, 120, 0.025, 0.18, 0.95, 85, 0)

        decision = heuristic_router().route(RouteRequest("Hello", Device.PHONE, telemetry))

        self.assertTrue(next(item for item in decision.candidates if item.device == Device.PHONE).eligible)
        self.assertTrue(next(item for item in decision.candidates if item.device == Device.PC).eligible)
        self.assertEqual(
            set(next(item for item in decision.candidates if item.device == Device.PHONE).penalties),
            {"latency", "energy", "quality"},
        )

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
        self.assertIsNone(result.metrics)
        self.assertNotIn("metrics", result.to_dict())

    def test_request_requires_all_three_telemetry_entries(self) -> None:
        with self.assertRaises(ValidationError):
            RouteRequest("Hello", Device.PHONE, {Device.PHONE: built_in_scenarios()["healthy"][Device.PHONE]})

    def test_telemetry_rejects_string_numbers_and_non_boolean_availability(self) -> None:
        with self.assertRaises(ValidationError):
            DeviceTelemetry(True, "10", 100, 0.01, 0.1, 0.1, 80, 0)
        with self.assertRaises(ValidationError):
            DeviceTelemetry("true", 10, 100, 0.01, 0.1, 0.1, 80, 0)

    def test_estimator_failure_mid_route_falls_back_to_static_behaviour(self) -> None:
        class FlakyEstimator:
            def estimate(self, prompt, intent=None):
                raise EstimatorUnavailableError("weights unloaded")

        router = heuristic_router(estimator=FlakyEstimator())

        decision = router.route(request_for("What's the weather tomorrow?"))

        # Same outcome as no estimator at all -- a runtime failure must not
        # take routing down or silently change the decision.
        self.assertEqual(decision.selected_device, Device.PHONE)
        self.assertFalse(decision.quality_degraded)

    def test_untrusted_estimate_forces_cloud_when_not_sensitive(self) -> None:
        class LowConfidenceEstimator:
            def estimate(self, prompt, intent=None):
                return _untrusted_estimate()

        router = heuristic_router(estimator=LowConfidenceEstimator())

        decision = router.route(request_for("What's the weather tomorrow?"))

        self.assertEqual(decision.selected_device, Device.CLOUD)
        self.assertTrue(decision.quality_degraded)
        self.assertIn("out of the calibration domain", decision.explanation)

    def test_untrusted_estimate_forces_pc_when_sensitive(self) -> None:
        class LowConfidenceEstimator:
            def estimate(self, prompt, intent=None):
                return _untrusted_estimate()

        router = heuristic_router(estimator=LowConfidenceEstimator())

        decision = router.route(request_for("Summarize records for alice@example.com"))

        self.assertEqual(decision.selected_device, Device.PC)
        self.assertTrue(decision.quality_degraded)
        self.assertIn("privacy-sensitive", decision.explanation)

    def test_untrusted_estimate_falls_back_when_forced_target_is_ineligible(self) -> None:
        class LowConfidenceEstimator:
            def estimate(self, prompt, intent=None):
                return _untrusted_estimate()

        router = heuristic_router(estimator=LowConfidenceEstimator())

        decision = router.route(request_for("What's the weather tomorrow?", scenario="cloud-offline"))

        self.assertNotEqual(decision.selected_device, Device.CLOUD)

    def test_trusted_estimate_is_unaffected_by_the_untrusted_fallback(self) -> None:
        class TrustedEstimator:
            quality_floor = 0.5

            def estimate(self, prompt, intent=None):
                return PromptEstimate(
                    p_pass={Device.PHONE: 0.9, Device.PC: 0.95, Device.CLOUD: 0.99},
                    length_p50={Device.PHONE: 20, Device.PC: 20, Device.CLOUD: 20},
                    length_p90={Device.PHONE: 30, Device.PC: 30, Device.CLOUD: 30},
                    confidence="high",
                    mean_distance=0.05,
                    neighbours=(),
                )

        router = heuristic_router(estimator=TrustedEstimator())

        decision = router.route(request_for("What's the weather tomorrow?"))

        self.assertNotIn("out of the calibration domain", decision.explanation)


if __name__ == "__main__":
    unittest.main()
