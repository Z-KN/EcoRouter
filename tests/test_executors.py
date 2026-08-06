import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import urllib.error

from ecorouter import (
    CirrascaleExecutor,
    CloudConfigurationError,
    CloudExecutionError,
    Device,
    EcoRouter,
    GenieXPhoneExecutor,
    HeuristicPromptAnalyzer,
    XEliteExecutor,
    OptimizationProfile,
    PcConfigurationError,
    PcExecutionError,
    PhoneConfigurationError,
    PhoneExecutionError,
    RouteRequest,
    build_executors,
    cirrascale_executors,
    hybrid_executors,
    x_elite_executors,
)
from ecorouter.executors import _ImagineBindings, phone_health
from ecorouter.scenarios import built_in_scenarios


class FakeResponse:
    def __init__(self, content="cloud response", *, model="Llama-3.1-8B", usage=None) -> None:
        self.first_content = content
        self.model = model
        self.usage = usage


class FakeClient:
    def __init__(self, models=None, content="cloud response", error=None, usage=None) -> None:
        self.models = models if models is not None else ["Llama-3.1-8B"]
        self.content = content
        self.error = error
        self.usage = usage
        self.catalog_calls = 0
        self.chat_calls = []
        self.model_type = None

    def get_available_models(self, model_type=None):
        self.catalog_calls += 1
        self.model_type = model_type
        if self.error == "catalog":
            raise RuntimeError("secret catalog details replacement-key")
        return self.models

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if self.error == "chat":
            raise RuntimeError("secret inference details replacement-key")
        return FakeResponse(self.content, usage=self.usage)


def cloud_decision():
    router = EcoRouter(analyzer=HeuristicPromptAnalyzer())
    request = RouteRequest(
        "What model are you?",
        Device.PC,
        built_in_scenarios()["healthy"],
        OptimizationProfile.HIGH_QUALITY,
    )
    decision = router.route(request)
    if decision.selected_device != Device.CLOUD:
        raise AssertionError("test fixture must route to cloud")
    return decision


def pc_decision():
    router = EcoRouter(analyzer=HeuristicPromptAnalyzer())
    request = RouteRequest(
        "What model are you?",
        Device.PC,
        built_in_scenarios()["healthy"],
        OptimizationProfile.LOW_LATENCY,
    )
    decision = router.route(request)
    if decision.selected_device != Device.PC:
        raise AssertionError("test fixture must route to pc")
    return decision


def phone_decision():
    router = EcoRouter(analyzer=HeuristicPromptAnalyzer())
    request = RouteRequest(
        "What model are you?",
        Device.PHONE,
        built_in_scenarios()["healthy"],
        OptimizationProfile.ENERGY_SAVER,
    )
    decision = router.route(request)
    if decision.selected_device != Device.PHONE:
        raise AssertionError("test fixture must route to phone")
    return decision


class CirrascaleExecutorTests(unittest.TestCase):
    def test_execute_observed_times_only_chat_and_extracts_sdk_usage(self) -> None:
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=11, total_tokens=18)
        client = FakeClient(usage=usage)
        ticks = iter((1_000_000_000, 2_234_567_000))
        executor = CirrascaleExecutor(
            client=client,
            message_factory=lambda **kwargs: kwargs,
            llm_model_type="llm",
            clock_ns=lambda: next(ticks),
        )

        observation = executor.execute_observed("What model are you?", cloud_decision())

        self.assertEqual(observation.response, "cloud response")
        self.assertEqual(observation.api_turnaround_latency_ms, 1234.567)
        self.assertEqual(observation.model_id, "Llama-3.1-8B")
        self.assertEqual(observation.prompt_tokens, 7)
        self.assertEqual(observation.completion_tokens, 11)
        self.assertEqual(observation.total_tokens, 18)
        self.assertEqual(client.catalog_calls, 1)
        self.assertEqual(len(client.chat_calls), 1)

    def test_missing_or_malformed_usage_does_not_discard_response(self) -> None:
        malformed = SimpleNamespace(
            prompt_tokens="7", completion_tokens=-1, total_tokens=18.5
        )
        for usage in (None, malformed):
            with self.subTest(usage=usage):
                executor = CirrascaleExecutor(
                    client=FakeClient(usage=usage),
                    message_factory=lambda **kwargs: kwargs,
                )
                observation = executor.execute_observed(
                    "What model are you?", cloud_decision()
                )

                self.assertEqual(observation.response, "cloud response")
                self.assertIsNone(observation.prompt_tokens)
                self.assertIsNone(observation.completion_tokens)
                self.assertIsNone(observation.total_tokens)

    def test_total_tokens_are_derived_when_sdk_total_is_missing(self) -> None:
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=None)
        executor = CirrascaleExecutor(
            client=FakeClient(usage=usage),
            message_factory=lambda **kwargs: kwargs,
        )

        observation = executor.execute_observed("What model are you?", cloud_decision())

        self.assertEqual(observation.total_tokens, 8)

    def test_execute_forwards_exact_request_and_caches_catalog(self) -> None:
        client = FakeClient(models=["z-model", "Llama-3.1-8B", "Llama-3.1-8B"])
        messages = []

        def message_factory(**kwargs):
            messages.append(kwargs)
            return kwargs

        executor = CirrascaleExecutor(
            client=client,
            message_factory=message_factory,
            llm_model_type="llm",
        )
        decision = cloud_decision()

        response = executor.execute("What model are you?", decision)
        models = executor.list_models()

        self.assertEqual(response, "cloud response")
        self.assertEqual(models, ("Llama-3.1-8B", "z-model"))
        self.assertEqual(client.catalog_calls, 1)
        self.assertEqual(client.model_type, "llm")
        self.assertEqual(messages, [{"role": "user", "content": "What model are you?"}])
        self.assertEqual(
            client.chat_calls,
            [
                {
                    "messages": [{"role": "user", "content": "What model are you?"}],
                    "model": "Llama-3.1-8B",
                    "max_tokens": decision.analysis.estimated_output_tokens,
                    "temperature": 0,
                }
            ],
        )

    def test_lazy_initialization_uses_secure_client_settings(self) -> None:
        captured = {}
        client = FakeClient()

        def client_factory(**kwargs):
            captured.update(kwargs)
            return client

        bindings = _ImagineBindings(client_factory, lambda **kwargs: kwargs, "llm")
        executor = CirrascaleExecutor(
            environ={
                "INFERENCE_CLOUD_API_KEY": "replacement-key",
                "INFERENCE_CLOUD_ENDPOINT": "https://aisuite.cirrascale.com/apis/v2/",
            }
        )

        with patch("ecorouter.executors._load_imagine_bindings", return_value=bindings):
            self.assertEqual(executor.list_models(), ("Llama-3.1-8B",))

        self.assertEqual(
            captured,
            {
                "endpoint": "https://aisuite.cirrascale.com/apis/v2",
                "api_key": "replacement-key",
                "max_retries": 1,
                "timeout": 60,
                "verify": True,
                "debug": False,
            },
        )

    def test_missing_environment_and_invalid_endpoint_fail_before_sdk_import(self) -> None:
        with patch("ecorouter.executors._load_imagine_bindings") as loader:
            with self.assertRaisesRegex(
                CloudConfigurationError, "INFERENCE_CLOUD_API_KEY"
            ):
                CirrascaleExecutor(environ={}).list_models()
            loader.assert_not_called()

        invalid_endpoints = (
            "http://aisuite.cirrascale.com/apis/v2",
            "https://user:password@aisuite.cirrascale.com/apis/v2",
            "https://aisuite.cirrascale.com/apis/v2?key=value",
        )
        for endpoint in invalid_endpoints:
            with self.subTest(endpoint=endpoint):
                executor = CirrascaleExecutor(
                    environ={
                        "INFERENCE_CLOUD_API_KEY": "replacement-key",
                        "INFERENCE_CLOUD_ENDPOINT": endpoint,
                    }
                )
                with self.assertRaisesRegex(CloudConfigurationError, "HTTPS URL"):
                    executor.list_models()

    def test_missing_sdk_failure_is_sanitized(self) -> None:
        executor = CirrascaleExecutor(
            environ={
                "INFERENCE_CLOUD_API_KEY": "private-key",
                "INFERENCE_CLOUD_ENDPOINT": "https://example.test/apis/v2",
            }
        )
        with patch(
            "ecorouter.executors._load_imagine_bindings",
            side_effect=CloudConfigurationError("cloud extra missing"),
        ):
            with self.assertRaisesRegex(CloudConfigurationError, "cloud extra missing") as caught:
                executor.list_models()

        self.assertNotIn("private-key", str(caught.exception))

    def test_unavailable_model_stops_before_inference(self) -> None:
        client = FakeClient(models=["another-model"])
        executor = CirrascaleExecutor(
            client=client,
            message_factory=lambda **kwargs: kwargs,
        )

        with self.assertRaisesRegex(CloudConfigurationError, "another-model"):
            executor.execute("What model are you?", cloud_decision())

        self.assertEqual(client.chat_calls, [])

    def test_sensitive_and_non_cloud_decisions_never_touch_client(self) -> None:
        client = FakeClient()
        executor = CirrascaleExecutor(
            client=client,
            message_factory=lambda **kwargs: kwargs,
        )
        decision = cloud_decision()
        sensitive = replace(
            decision,
            analysis=replace(decision.analysis, sensitive=True, pii_categories=("email",)),
        )
        local = replace(decision, selected_device=Device.PC, model_id="pc-model")

        with self.assertRaisesRegex(CloudExecutionError, "Privacy policy"):
            executor.execute("alice@example.com", sensitive)
        with self.assertRaisesRegex(CloudExecutionError, "cloud routing decision"):
            executor.execute("What model are you?", local)

        self.assertEqual(client.catalog_calls, 0)
        self.assertEqual(client.chat_calls, [])

    def test_provider_failures_and_empty_responses_are_sanitized(self) -> None:
        private_value = "alice@example.com"
        key_value = "replacement-key"
        for client, expected in (
            (FakeClient(error="catalog"), "model discovery failed"),
            (FakeClient(error="chat"), "inference failed"),
            (FakeClient(content="   "), "empty response"),
        ):
            with self.subTest(expected=expected):
                executor = CirrascaleExecutor(
                    environ={
                        "INFERENCE_CLOUD_API_KEY": key_value,
                        "INFERENCE_CLOUD_ENDPOINT": "https://example.test/apis/v2",
                    },
                    client=client,
                    message_factory=lambda **kwargs: kwargs,
                )
                with self.assertRaisesRegex(CloudExecutionError, expected) as caught:
                    executor.execute(private_value, cloud_decision())
                serialized = json.dumps({"error": str(caught.exception)})
                self.assertNotIn(private_value, serialized)
                self.assertNotIn(key_value, serialized)
                self.assertNotIn("secret", serialized)

    def test_combined_factory_keeps_local_simulators_and_live_cloud(self) -> None:
        cloud = CirrascaleExecutor(client=FakeClient(), message_factory=lambda **kwargs: kwargs)
        executors = cirrascale_executors(cloud)

        self.assertIs(executors[Device.CLOUD], cloud)
        self.assertEqual(set(executors), set(Device))
        self.assertEqual(executors[Device.PHONE].device, Device.PHONE)
        self.assertEqual(executors[Device.PC].device, Device.PC)


class XEliteExecutorTests(unittest.TestCase):
    def test_execute_observed_posts_prompt_and_parses_openai_shape(self) -> None:
        captured = {}

        def http_post(url, payload):
            captured["url"] = url
            captured["payload"] = payload
            return {
                "choices": [{"message": {"content": "I am Qwen3-VL-4B-Instruct."}}],
                "model": "ai-hub-models/Qwen3-VL-4B-Instruct",
                "usage": {"prompt_tokens": 30, "completion_tokens": 22, "total_tokens": 52},
            }

        ticks = iter((1_000_000_000, 1_093_000_000))
        executor = XEliteExecutor(
            endpoint="http://localhost:8000",
            http_post=http_post,
            clock_ns=lambda: next(ticks),
        )
        decision = pc_decision()

        observation = executor.execute_observed("What model are you?", decision)

        self.assertEqual(captured["url"], "http://localhost:8000/v1/chat/completions")
        self.assertEqual(
            captured["payload"]["messages"],
            [{"role": "user", "content": "What model are you?"}],
        )
        self.assertEqual(
            captured["payload"]["max_tokens"], decision.analysis.estimated_output_tokens
        )
        self.assertEqual(observation.response, "I am Qwen3-VL-4B-Instruct.")
        self.assertEqual(observation.api_turnaround_latency_ms, 93.0)
        self.assertEqual(observation.model_id, "ai-hub-models/Qwen3-VL-4B-Instruct")
        self.assertEqual(observation.prompt_tokens, 30)
        self.assertEqual(observation.completion_tokens, 22)
        self.assertEqual(observation.total_tokens, 52)

    def test_execute_forwards_exact_prompt(self) -> None:
        executor = XEliteExecutor(
            http_post=lambda url, payload: {"choices": [{"message": {"content": "hi"}}]}
        )

        self.assertEqual(executor.execute("What model are you?", pc_decision()), "hi")

    def test_endpoint_defaults_and_reads_environment_override(self) -> None:
        captured = {}

        def http_post(url, payload):
            captured["url"] = url
            return {"choices": [{"message": {"content": "hi"}}]}

        executor = XEliteExecutor(
            environ={"XELITE_SERVER_ENDPOINT": "http://192.168.1.214:9000/"},
            http_post=http_post,
        )

        executor.execute_observed("What model are you?", pc_decision())

        self.assertEqual(captured["url"], "http://192.168.1.214:9000/v1/chat/completions")

    def test_missing_or_malformed_usage_does_not_discard_response(self) -> None:
        for body in (
            {"choices": [{"message": {"content": "hi"}}]},
            {"choices": [{"message": {"content": "hi"}}], "usage": "not-a-dict"},
        ):
            with self.subTest(body=body):
                executor = XEliteExecutor(http_post=lambda url, payload, body=body: body)
                observation = executor.execute_observed("What model are you?", pc_decision())

                self.assertEqual(observation.response, "hi")
                self.assertIsNone(observation.prompt_tokens)
                self.assertIsNone(observation.completion_tokens)
                self.assertIsNone(observation.total_tokens)

    def test_non_pc_decision_is_rejected_before_any_request(self) -> None:
        calls = []
        executor = XEliteExecutor(
            http_post=lambda url, payload: calls.append(1)
            or {"choices": [{"message": {"content": "hi"}}]}
        )

        with self.assertRaisesRegex(PcExecutionError, "PC routing decision"):
            executor.execute("What model are you?", cloud_decision())

        self.assertEqual(calls, [])

    def test_unreachable_server_raises_configuration_error(self) -> None:
        def http_post(url, payload):
            raise urllib.error.URLError("connection refused")

        executor = XEliteExecutor(endpoint="http://localhost:8000", http_post=http_post)

        with self.assertRaisesRegex(PcConfigurationError, "could not reach"):
            executor.execute("What model are you?", pc_decision())

    def test_malformed_and_empty_responses_raise_execution_error(self) -> None:
        for body, expected in (
            ({"unexpected": "shape"}, "unexpected response shape"),
            ({"choices": []}, "unexpected response shape"),
            ({"choices": [{"message": {"content": "   "}}]}, "empty response"),
        ):
            with self.subTest(expected=expected):
                executor = XEliteExecutor(http_post=lambda url, payload, body=body: body)
                with self.assertRaisesRegex(PcExecutionError, expected):
                    executor.execute("What model are you?", pc_decision())

    def test_x_elite_executors_keeps_phone_and_cloud_simulated(self) -> None:
        pc = XEliteExecutor(http_post=lambda url, payload: {"choices": [{"message": {"content": "hi"}}]})
        executors = x_elite_executors(pc)

        self.assertIs(executors[Device.PC], pc)
        self.assertEqual(set(executors), set(Device))
        self.assertEqual(executors[Device.PHONE].device, Device.PHONE)
        self.assertEqual(executors[Device.CLOUD].device, Device.CLOUD)

    def test_hybrid_executors_makes_both_pc_and_cloud_live(self) -> None:
        pc = XEliteExecutor(http_post=lambda url, payload: {"choices": [{"message": {"content": "hi"}}]})
        cloud = CirrascaleExecutor(client=FakeClient(), message_factory=lambda **kwargs: kwargs)
        executors = hybrid_executors(cloud_executor=cloud, pc_executor=pc)

        self.assertIs(executors[Device.PC], pc)
        self.assertIs(executors[Device.CLOUD], cloud)
        self.assertEqual(executors[Device.PHONE].device, Device.PHONE)


class GenieXPhoneExecutorTests(unittest.TestCase):
    def test_execute_observed_posts_prompt_and_parses_phone_profile(self) -> None:
        captured = {}

        def http_post(url, payload):
            captured["url"] = url
            captured["payload"] = payload
            return {
                "choices": [{"message": {"content": "I am Qwen3-0.6B-GGUF."}}],
                "model": "Qwen3-0.6B-GGUF",
                "usage": {"prompt_tokens": 12, "completion_tokens": 40, "total_tokens": 52},
                "phone_profile": {
                    "ttft_ms": 88.5,
                    "prefill_speed_tok_s": 140.2,
                    "decode_speed_tok_s": 18.6,
                    "compute_unit": "npu",
                    "measured_energy_mj": 512.0,
                    "tokens_per_joule": 78.1,
                    "energy_available": True,
                    "charging": False,
                },
            }

        ticks = iter((1_000_000_000, 1_410_000_000))
        executor = GenieXPhoneExecutor(
            endpoint="http://192.168.1.50:8080",
            token="secret-token",
            http_post=http_post,
            clock_ns=lambda: next(ticks),
        )
        decision = phone_decision()

        observation = executor.execute_observed("What model are you?", decision)

        self.assertEqual(captured["url"], "http://192.168.1.50:8080/v1/chat/completions")
        self.assertEqual(
            captured["payload"]["messages"], [{"role": "user", "content": "What model are you?"}]
        )
        self.assertEqual(
            captured["payload"]["max_tokens"], decision.analysis.estimated_output_tokens
        )
        self.assertEqual(observation.response, "I am Qwen3-0.6B-GGUF.")
        self.assertEqual(observation.api_turnaround_latency_ms, 410.0)
        self.assertEqual(observation.model_id, "Qwen3-0.6B-GGUF")
        self.assertEqual(observation.prompt_tokens, 12)
        self.assertEqual(observation.completion_tokens, 40)
        self.assertEqual(observation.total_tokens, 52)
        self.assertEqual(observation.ttft_ms, 88.5)
        self.assertEqual(observation.prefill_speed_tokens_per_second, 140.2)
        self.assertEqual(observation.decode_speed_tokens_per_second, 18.6)
        self.assertEqual(observation.compute_unit, "npu")
        self.assertEqual(observation.measured_energy_joules, 0.512)
        self.assertEqual(observation.tokens_per_joule, 78.1)

    def test_energy_is_only_trusted_when_phone_reports_it_available(self) -> None:
        def http_post(url, payload):
            return {
                "choices": [{"message": {"content": "hi"}}],
                "phone_profile": {
                    "measured_energy_mj": 512.0,
                    # e.g. charging, or a non-NPU / uncalibrated compute unit.
                    "energy_available": False,
                },
            }

        executor = GenieXPhoneExecutor(endpoint="http://phone:8080", token="t", http_post=http_post)

        observation = executor.execute_observed("What model are you?", phone_decision())

        self.assertIsNone(observation.measured_energy_joules)

    def test_missing_phone_profile_leaves_stats_none(self) -> None:
        executor = GenieXPhoneExecutor(
            endpoint="http://phone:8080",
            token="t",
            http_post=lambda url, payload: {"choices": [{"message": {"content": "hi"}}]},
        )

        observation = executor.execute_observed("What model are you?", phone_decision())

        self.assertEqual(observation.response, "hi")
        self.assertIsNone(observation.ttft_ms)
        self.assertIsNone(observation.measured_energy_joules)
        self.assertIsNone(observation.compute_unit)

    def test_execute_forwards_exact_prompt(self) -> None:
        executor = GenieXPhoneExecutor(
            endpoint="http://phone:8080",
            token="t",
            http_post=lambda url, payload: {"choices": [{"message": {"content": "hi"}}]},
        )

        self.assertEqual(executor.execute("What model are you?", phone_decision()), "hi")

    def test_endpoint_and_token_read_from_environment(self) -> None:
        captured = {}

        def http_post(url, payload):
            captured["url"] = url
            return {"choices": [{"message": {"content": "hi"}}]}

        executor = GenieXPhoneExecutor(
            environ={
                "PHONE_SERVER_ENDPOINT": "http://192.168.1.77:8080/",
                "PHONE_SERVER_TOKEN": "env-token",
            },
            http_post=http_post,
        )

        executor.execute_observed("What model are you?", phone_decision())

        self.assertEqual(captured["url"], "http://192.168.1.77:8080/v1/chat/completions")

    def test_missing_endpoint_or_token_raise_configuration_error_before_any_request(self) -> None:
        calls = []
        no_endpoint = GenieXPhoneExecutor(
            token="t", environ={}, http_post=lambda url, payload: calls.append(1)
        )
        with self.assertRaisesRegex(PhoneConfigurationError, "PHONE_SERVER_ENDPOINT"):
            no_endpoint.execute("hi", phone_decision())

        no_token = GenieXPhoneExecutor(
            endpoint="http://phone:8080", environ={}, http_post=lambda url, payload: calls.append(1)
        )
        with self.assertRaisesRegex(PhoneConfigurationError, "PHONE_SERVER_TOKEN"):
            no_token.execute("hi", phone_decision())

        self.assertEqual(calls, [])

    def test_non_phone_decision_is_rejected_before_any_request(self) -> None:
        calls = []
        executor = GenieXPhoneExecutor(
            endpoint="http://phone:8080",
            token="t",
            http_post=lambda url, payload: calls.append(1)
            or {"choices": [{"message": {"content": "hi"}}]},
        )

        with self.assertRaisesRegex(PhoneExecutionError, "phone routing decision"):
            executor.execute("What model are you?", cloud_decision())

        self.assertEqual(calls, [])

    def test_unreachable_server_raises_configuration_error(self) -> None:
        def http_post(url, payload):
            raise urllib.error.URLError("connection refused")

        executor = GenieXPhoneExecutor(endpoint="http://phone:8080", token="t", http_post=http_post)

        with self.assertRaisesRegex(PhoneConfigurationError, "could not reach"):
            executor.execute("What model are you?", phone_decision())

    def test_malformed_and_empty_responses_raise_execution_error(self) -> None:
        for body, expected in (
            ({"unexpected": "shape"}, "unexpected response shape"),
            ({"choices": []}, "unexpected response shape"),
            ({"choices": [{"message": {"content": "   "}}]}, "empty response"),
        ):
            with self.subTest(expected=expected):
                executor = GenieXPhoneExecutor(
                    endpoint="http://phone:8080", token="t", http_post=lambda url, payload, body=body: body
                )
                with self.assertRaisesRegex(PhoneExecutionError, expected):
                    executor.execute("What model are you?", phone_decision())

    def test_live_http_post_sends_bearer_token(self) -> None:
        captured = {}

        def fake_default_http_post(url, payload, *, timeout_seconds, headers=None):
            captured["url"] = url
            captured["headers"] = headers
            return {"choices": [{"message": {"content": "hi"}}]}

        executor = GenieXPhoneExecutor(endpoint="http://phone:8080", token="secret-token")
        with patch("ecorouter.executors._default_http_post", fake_default_http_post):
            executor.execute_observed("What model are you?", phone_decision())

        self.assertEqual(captured["headers"], {"Authorization": "Bearer secret-token"})


class BuildExecutorsTests(unittest.TestCase):
    def test_no_flags_keeps_everything_simulated(self) -> None:
        executors = build_executors()

        self.assertEqual(set(executors), set(Device))
        for device in Device:
            self.assertEqual(executors[device].device, device)

    def test_live_phone_only(self) -> None:
        phone = GenieXPhoneExecutor(endpoint="http://phone:8080", token="t")
        executors = build_executors(live_phone=True, phone_executor=phone)

        self.assertIs(executors[Device.PHONE], phone)
        self.assertEqual(executors[Device.PC].device, Device.PC)
        self.assertEqual(executors[Device.CLOUD].device, Device.CLOUD)

    def test_all_three_live(self) -> None:
        phone = GenieXPhoneExecutor(endpoint="http://phone:8080", token="t")
        pc = XEliteExecutor(http_post=lambda url, payload: {"choices": [{"message": {"content": "hi"}}]})
        cloud = CirrascaleExecutor(client=FakeClient(), message_factory=lambda **kwargs: kwargs)
        executors = build_executors(
            live_phone=True,
            live_pc=True,
            live_cloud=True,
            phone_executor=phone,
            pc_executor=pc,
            cloud_executor=cloud,
        )

        self.assertIs(executors[Device.PHONE], phone)
        self.assertIs(executors[Device.PC], pc)
        self.assertIs(executors[Device.CLOUD], cloud)


class PhoneHealthTests(unittest.TestCase):
    def test_returns_parsed_json_from_health_endpoint(self) -> None:
        captured = {}

        def http_get(url):
            captured["url"] = url
            return {"status": "healthy", "model": "Qwen3-0.6B-GGUF"}

        result = phone_health(endpoint="http://192.168.1.50:8080", http_get=http_get)

        self.assertEqual(captured["url"], "http://192.168.1.50:8080/health")
        self.assertEqual(result["status"], "healthy")

    def test_missing_endpoint_raises_configuration_error(self) -> None:
        with self.assertRaisesRegex(PhoneConfigurationError, "PHONE_SERVER_ENDPOINT"):
            phone_health(environ={})

    def test_unreachable_server_raises_configuration_error(self) -> None:
        def http_get(url):
            raise urllib.error.URLError("connection refused")

        with self.assertRaisesRegex(PhoneConfigurationError, "could not reach"):
            phone_health(endpoint="http://phone:8080", http_get=http_get)


if __name__ == "__main__":
    unittest.main()
