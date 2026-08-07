import contextlib
import io
import json
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from peqrouter.cli import HEADS_DIR, main
from peqrouter.estimator import EstimatorUnavailableError
from peqrouter.executors import default_simulated_executors
from peqrouter.models import (
    CloudConfigurationError,
    Device,
    DeviceConfig,
    ExecutionObservation,
    PrivacyInitializationError,
)
from peqrouter.scenarios import built_in_scenarios


def _flat_configs():
    # Equal capability scores isolate these dispatch tests from the quality
    # penalty -- phone's lower static capability_score (0.60 vs pc's 0.80)
    # can otherwise outweigh its latency/energy edge, since the profile
    # weights now only cover latency/energy/quality.
    return {device: DeviceConfig(f"{device.value}-model", 0.80) for device in Device}


def _pc_forced_scenarios():
    # Phone/cloud made unavailable in the "healthy" scenario so PC wins by
    # elimination -- PC is genuinely slower and more energy-hungry per token
    # than phone on the real calibrated constants (see scenarios.py), so it
    # no longer wins latency/energy tradeoffs naturally the way the old
    # illustrative dummy numbers made it appear to.
    scenarios = dict(built_in_scenarios())
    healthy = dict(scenarios["healthy"])
    healthy[Device.PHONE] = replace(healthy[Device.PHONE], available=False)
    healthy[Device.CLOUD] = replace(healthy[Device.CLOUD], available=False)
    scenarios["healthy"] = healthy
    return scenarios


class LiveCloudStub:
    def execute(self, prompt, decision):
        return "live cloud response"

    def execute_observed(self, prompt, decision):
        return ExecutionObservation(
            response="live cloud response",
            api_turnaround_latency_ms=321.9876,
            model_id=decision.model_id,
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
        )


class LivePcStub:
    def execute(self, prompt, decision):
        return "live pc response"

    def execute_observed(self, prompt, decision):
        return ExecutionObservation(
            response="live pc response",
            api_turnaround_latency_ms=93.0,
            model_id=decision.model_id,
            prompt_tokens=30,
            completion_tokens=22,
            total_tokens=52,
        )


class LiveMeasuredPcStub:
    def execute(self, prompt, decision):
        return "live pc response"

    def execute_observed(self, prompt, decision):
        return ExecutionObservation(
            response="live pc response",
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


class LivePhoneStub:
    def execute(self, prompt, decision):
        return "live phone response"

    def execute_observed(self, prompt, decision):
        return ExecutionObservation(
            response="live phone response",
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


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate ordinary router-mechanics tests from the real calibrated
        # estimator (heavy, and would change which device several fixture
        # prompts route to). Tests that specifically exercise the estimator
        # wiring override this patch themselves.
        patcher = patch(
            "peqrouter.cli.CalibratedEstimator",
            side_effect=EstimatorUnavailableError("stub: no estimator in unit tests"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

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
                "peqrouter.cli.PEQRouter",
                side_effect=PrivacyInitializationError("privacy runtime unavailable"),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(["route", "--origin", "phone", "--prompt", "Hello"])

        self.assertEqual(code, 5)
        self.assertEqual(
            stderr.getvalue().strip().splitlines()[-1],
            "privacy error: privacy runtime unavailable",
        )

    def test_cloud_models_emits_human_and_json_output(self) -> None:
        fake_executor = Mock()
        fake_executor.list_models.return_value = ("Llama-3.1-8B", "other-model")
        with patch("peqrouter.cli.CirrascaleExecutor", return_value=fake_executor):
            human = io.StringIO()
            with contextlib.redirect_stdout(human):
                human_code = main(["cloud-models"])
            machine = io.StringIO()
            with contextlib.redirect_stdout(machine):
                machine_code = main(["cloud-models", "--json"])

        self.assertEqual(human_code, 0)
        self.assertIn("Available Cirrascale LLMs: 2", human.getvalue())
        self.assertEqual(machine_code, 0)
        self.assertEqual(
            json.loads(machine.getvalue()),
            {"count": 2, "models": ["Llama-3.1-8B", "other-model"]},
        )

    def test_cloud_configuration_failure_returns_exit_code_four(self) -> None:
        stderr = io.StringIO()
        with (
            patch(
                "peqrouter.cli.CirrascaleExecutor",
                side_effect=CloudConfigurationError("cloud configuration unavailable"),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(["cloud-models"])

        self.assertEqual(code, 4)
        self.assertEqual(
            stderr.getvalue().strip(),
            "execution error: cloud configuration unavailable",
        )

    def test_live_cloud_flag_dispatches_cloud_route(self) -> None:
        executors = default_simulated_executors()
        executors[Device.CLOUD] = LiveCloudStub()
        stdout = io.StringIO()
        with (
            patch("peqrouter.cli.build_executors", return_value=executors) as factory,
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--origin",
                    "pc",
                    "--prompt",
                    "What model are you?",
                    "--profile",
                    "high-quality",
                    "--live-cloud",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["decision"]["selected_device"], "cloud")
        self.assertEqual(payload["response"], "live cloud response")
        self.assertEqual(payload["metrics"]["api_turnaround_latency_ms"], 321.988)
        self.assertEqual(payload["metrics"]["total_tokens"], 15)
        self.assertIsNone(payload["metrics"]["measured_energy_joules"])
        self.assertEqual(payload["metrics"]["estimated_energy_joules"], 33.712916)
        self.assertEqual(payload["metrics"]["confidence"], "uncalibrated")
        factory.assert_called_once_with(live_phone=False, live_pc=False, live_cloud=True)

    def test_live_cloud_human_output_labels_observations_and_estimate(self) -> None:
        executors = default_simulated_executors()
        executors[Device.CLOUD] = LiveCloudStub()
        stdout = io.StringIO()
        with (
            patch("peqrouter.cli.build_executors", return_value=executors),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--origin",
                    "pc",
                    "--prompt",
                    "What model are you?",
                    "--profile",
                    "high-quality",
                    "--live-cloud",
                ]
            )

        output = stdout.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("API turnaround latency: 321.988 ms", output)
        self.assertIn("Measured energy: unavailable", output)
        self.assertIn("Estimated energy: 33.712916 J", output)
        self.assertIn("uncalibrated", output)

    def test_live_cloud_flag_keeps_local_route_simulated(self) -> None:
        executors = default_simulated_executors()
        executors[Device.CLOUD] = LiveCloudStub()
        stdout = io.StringIO()
        with (
            patch("peqrouter.cli.build_executors", return_value=executors),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--origin",
                    "phone",
                    "--prompt",
                    "What's the weather?",
                    "--live-cloud",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["decision"]["selected_device"], "phone")
        self.assertIn("Simulated", payload["response"])

    def test_live_pc_flag_dispatches_pc_route(self) -> None:
        executors = default_simulated_executors()
        executors[Device.PC] = LivePcStub()
        stdout = io.StringIO()
        with (
            patch("peqrouter.cli.build_executors", return_value=executors) as factory,
            patch("peqrouter.cli.built_in_scenarios", return_value=_pc_forced_scenarios()),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--origin",
                    "pc",
                    "--prompt",
                    "What model are you?",
                    "--profile",
                    "low-latency",
                    "--live-pc",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["decision"]["selected_device"], "pc")
        self.assertEqual(payload["response"], "live pc response")
        self.assertEqual(payload["metrics"]["api_turnaround_latency_ms"], 93.0)
        self.assertEqual(payload["metrics"]["total_tokens"], 52)
        factory.assert_called_once_with(live_phone=False, live_pc=True, live_cloud=False)

    def test_live_pc_flag_keeps_other_routes_simulated(self) -> None:
        executors = default_simulated_executors()
        executors[Device.PC] = LivePcStub()
        stdout = io.StringIO()
        with (
            patch("peqrouter.cli.build_executors", return_value=executors),
            patch("peqrouter.cli.load_device_configs", return_value=_flat_configs()),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--origin",
                    "phone",
                    "--prompt",
                    "What's the weather?",
                    "--config",
                    "unused",
                    "--live-pc",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["decision"]["selected_device"], "phone")
        self.assertIn("Simulated", payload["response"])

    def test_live_pc_flag_dispatches_pc_route_with_full_stats(self) -> None:
        executors = default_simulated_executors()
        executors[Device.PC] = LiveMeasuredPcStub()
        stdout = io.StringIO()
        with (
            patch("peqrouter.cli.build_executors", return_value=executors) as factory,
            patch("peqrouter.cli.built_in_scenarios", return_value=_pc_forced_scenarios()),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--origin",
                    "pc",
                    "--prompt",
                    "What model are you?",
                    "--profile",
                    "low-latency",
                    "--live-pc",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["decision"]["selected_device"], "pc")
        self.assertEqual(payload["response"], "live pc response")
        metrics = payload["metrics"]
        self.assertAlmostEqual(metrics["measured_energy_joules"], 9.1467)
        self.assertEqual(metrics["confidence"], "measured")
        self.assertIn("measured whole-laptop battery discharge", metrics["energy_scope"])
        self.assertEqual(metrics["ttft_ms"], 45.3)
        self.assertEqual(metrics["prefill_speed_tokens_per_second"], 900.0)
        self.assertEqual(metrics["decode_speed_tokens_per_second"], 21.0)
        self.assertEqual(metrics["tokens_per_joule"], 2.4)
        self.assertEqual(metrics["compute_unit"], "npu")
        factory.assert_called_once_with(live_phone=False, live_pc=True, live_cloud=False)

    def test_live_pc_human_output_shows_measured_energy_and_throughput(self) -> None:
        executors = default_simulated_executors()
        executors[Device.PC] = LiveMeasuredPcStub()
        stdout = io.StringIO()
        with (
            patch("peqrouter.cli.build_executors", return_value=executors),
            patch("peqrouter.cli.built_in_scenarios", return_value=_pc_forced_scenarios()),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--origin",
                    "pc",
                    "--prompt",
                    "What model are you?",
                    "--profile",
                    "low-latency",
                    "--live-pc",
                ]
            )

        output = stdout.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Measured energy: 9.146700 J", output)
        self.assertIn("TTFT: 45.30 ms", output)

    def test_live_cloud_and_live_pc_together_pass_both_flags(self) -> None:
        executors = default_simulated_executors()
        executors[Device.CLOUD] = LiveCloudStub()
        executors[Device.PC] = LivePcStub()
        stdout = io.StringIO()
        with (
            patch("peqrouter.cli.build_executors", return_value=executors) as factory,
            patch("peqrouter.cli.built_in_scenarios", return_value=_pc_forced_scenarios()),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--origin",
                    "pc",
                    "--prompt",
                    "What model are you?",
                    "--profile",
                    "low-latency",
                    "--live-cloud",
                    "--live-pc",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["decision"]["selected_device"], "pc")
        self.assertEqual(payload["response"], "live pc response")
        factory.assert_called_once_with(live_phone=False, live_pc=True, live_cloud=True)

    def test_live_phone_flag_dispatches_phone_route_with_full_stats(self) -> None:
        executors = default_simulated_executors()
        executors[Device.PHONE] = LivePhoneStub()
        stdout = io.StringIO()
        with (
            patch("peqrouter.cli.build_executors", return_value=executors) as factory,
            patch("peqrouter.cli.load_device_configs", return_value=_flat_configs()),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--origin",
                    "phone",
                    "--prompt",
                    "What model are you?",
                    "--profile",
                    "low-energy",
                    "--config",
                    "unused",
                    "--live-phone",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["decision"]["selected_device"], "phone")
        self.assertEqual(payload["response"], "live phone response")
        metrics = payload["metrics"]
        self.assertEqual(metrics["measured_energy_joules"], 0.512)
        self.assertEqual(metrics["confidence"], "measured")
        self.assertIn("measured whole-device battery discharge", metrics["energy_scope"])
        self.assertEqual(metrics["ttft_ms"], 88.5)
        self.assertEqual(metrics["prefill_speed_tokens_per_second"], 140.2)
        self.assertEqual(metrics["decode_speed_tokens_per_second"], 18.6)
        self.assertEqual(metrics["tokens_per_joule"], 78.1)
        self.assertEqual(metrics["compute_unit"], "npu")
        factory.assert_called_once_with(live_phone=True, live_pc=False, live_cloud=False)

    def test_live_phone_human_output_shows_measured_energy_and_throughput(self) -> None:
        executors = default_simulated_executors()
        executors[Device.PHONE] = LivePhoneStub()
        stdout = io.StringIO()
        with (
            patch("peqrouter.cli.build_executors", return_value=executors),
            patch("peqrouter.cli.load_device_configs", return_value=_flat_configs()),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--origin",
                    "phone",
                    "--prompt",
                    "What model are you?",
                    "--config",
                    "unused",
                    "--live-phone",
                ]
            )

        output = stdout.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Measured energy: 0.512000 J", output)
        self.assertIn("TTFT: 88.50 ms", output)
        self.assertIn("Prefill speed: 140.20 tok/s", output)
        self.assertIn("Decode speed: 18.60 tok/s", output)
        self.assertIn("Efficiency: 78.10 tok/J", output)
        self.assertIn("Compute unit: npu", output)

    def test_live_phone_flag_keeps_other_routes_simulated(self) -> None:
        executors = default_simulated_executors()
        executors[Device.PHONE] = LivePhoneStub()
        stdout = io.StringIO()
        with (
            patch("peqrouter.cli.build_executors", return_value=executors),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--origin",
                    "pc",
                    "--prompt",
                    "What model are you?",
                    "--profile",
                    "high-quality",
                    "--live-phone",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertNotEqual(payload["decision"]["selected_device"], "phone")
        self.assertIn("Simulated", payload["response"])

    def test_phone_health_emits_human_and_json_output(self) -> None:
        with patch(
            "peqrouter.cli.phone_health",
            return_value={"status": "healthy", "model": "Qwen3-0.6B-GGUF", "uptime_s": 12.3, "requests_served": 4},
        ):
            human = io.StringIO()
            with contextlib.redirect_stdout(human):
                human_code = main(["phone-health"])
            machine = io.StringIO()
            with contextlib.redirect_stdout(machine):
                machine_code = main(["phone-health", "--json"])

        self.assertEqual(human_code, 0)
        self.assertIn("Phone server status: healthy", human.getvalue())
        self.assertIn("Model: Qwen3-0.6B-GGUF", human.getvalue())
        self.assertEqual(machine_code, 0)
        self.assertEqual(
            json.loads(machine.getvalue()),
            {"status": "healthy", "model": "Qwen3-0.6B-GGUF", "uptime_s": 12.3, "requests_served": 4},
        )

    def test_phone_health_configuration_failure_returns_exit_code_four(self) -> None:
        from peqrouter.models import PhoneConfigurationError

        stderr = io.StringIO()
        with (
            patch(
                "peqrouter.cli.phone_health",
                side_effect=PhoneConfigurationError("missing required environment variable PHONE_SERVER_ENDPOINT."),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(["phone-health"])

        self.assertEqual(code, 4)
        self.assertIn("PHONE_SERVER_ENDPOINT", stderr.getvalue())

    def test_no_estimator_flag_skips_estimator_construction(self) -> None:
        with patch("peqrouter.cli.CalibratedEstimator") as estimator_cls:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["route", "--origin", "phone", "--prompt", "Hello", "--no-estimator"])

        self.assertEqual(code, 0)
        estimator_cls.assert_not_called()

    def test_estimator_construction_failure_falls_back_with_a_stderr_notice(self) -> None:
        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            patch(
                "peqrouter.cli.CalibratedEstimator",
                side_effect=EstimatorUnavailableError("heads not cached"),
            ),
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(stdout),
        ):
            code = main(["route", "--origin", "phone", "--prompt", "Hello"])

        self.assertEqual(code, 0)
        self.assertIn(
            "routing without the calibrated estimator: heads not cached", stderr.getvalue()
        )
        self.assertIn("Selected:", stdout.getvalue())

    def test_calibrated_estimator_is_wired_into_the_router(self) -> None:
        sentinel_estimator = object()
        with (
            patch(
                "peqrouter.cli.CalibratedEstimator", return_value=sentinel_estimator
            ) as estimator_cls,
            patch(
                "peqrouter.cli.PEQRouter", side_effect=RuntimeError("stop after construction")
            ) as router_cls,
        ):
            with self.assertRaises(RuntimeError):
                main(["route", "--origin", "phone", "--prompt", "Hello"])

        estimator_cls.assert_called_once_with(HEADS_DIR)
        router_cls.assert_called_once_with(None, estimator=sentinel_estimator)


if __name__ == "__main__":
    unittest.main()
