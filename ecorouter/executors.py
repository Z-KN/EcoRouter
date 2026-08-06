"""Execution adapter contracts, simulators, and optional live cloud/PC support."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from time import perf_counter_ns
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from .models import (
    CLOUD_AI_100_TDP_WATTS,
    CloudConfigurationError,
    CloudExecutionError,
    Device,
    ExecutionObservation,
    PcConfigurationError,
    PcExecutionError,
    PhoneConfigurationError,
    PhoneExecutionError,
    RouteDecision,
)


_API_KEY_ENV = "INFERENCE_CLOUD_API_KEY"
_ENDPOINT_ENV = "INFERENCE_CLOUD_ENDPOINT"

_XELITE_ENDPOINT_ENV = "XELITE_SERVER_ENDPOINT"
_DEFAULT_XELITE_ENDPOINT = "http://localhost:8000"

_PHONE_ENDPOINT_ENV = "PHONE_SERVER_ENDPOINT"
_PHONE_TOKEN_ENV = "PHONE_SERVER_TOKEN"


@dataclass(frozen=True)
class _ImagineBindings:
    client_factory: Callable[..., Any]
    message_factory: Callable[..., Any]
    llm_model_type: Any


def _load_imagine_bindings() -> _ImagineBindings:
    try:
        from imagine import ChatMessage, ImagineClient, ModelType
    except (ImportError, ModuleNotFoundError):
        raise CloudConfigurationError(
            "Cirrascale cloud support is not installed; install EcoRouter with the 'cloud' extra."
        ) from None
    except Exception:
        raise CloudConfigurationError("Cirrascale cloud support could not initialize.") from None

    return _ImagineBindings(ImagineClient, ChatMessage, ModelType.LLM)


class Executor(Protocol):
    def execute(self, prompt: str, decision: RouteDecision) -> str:
        """Execute a prompt on the selected device and return its response."""


@runtime_checkable
class ObservedExecutor(Protocol):
    def execute_observed(self, prompt: str, decision: RouteDecision) -> ExecutionObservation:
        """Execute once and return response plus optional provider observations."""


class SimulatedExecutor:
    """Return a deterministic receipt without exposing or processing prompt text."""

    def __init__(self, device: Device) -> None:
        self.device = device

    def execute(self, prompt: str, decision: RouteDecision) -> str:
        if decision.selected_device != self.device:
            raise ValueError("executor device does not match the routing decision")
        total_tokens = (
            decision.analysis.estimated_input_tokens + decision.analysis.estimated_output_tokens
        )
        return (
            f"Simulated {decision.model_id} execution on {self.device.value} accepted a "
            f"{decision.analysis.intent.value} request ({total_tokens} estimated tokens)."
        )


class CirrascaleExecutor:
    """Invoke a privacy-approved cloud decision through Cirrascale Imagine."""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        client: Any | None = None,
        message_factory: Callable[..., Any] | None = None,
        llm_model_type: Any | None = None,
        timeout_seconds: float = 60,
        max_retries: int = 1,
        clock_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self._environ = environ if environ is not None else os.environ
        self._client = client
        self._message_factory = message_factory
        self._llm_model_type = llm_model_type
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._clock_ns = clock_ns
        self._available_models: tuple[str, ...] | None = None

    def list_models(self) -> tuple[str, ...]:
        """Return a cached, deterministic list of available Cirrascale LLMs."""

        if self._available_models is not None:
            return self._available_models

        client = self._ensure_client()
        try:
            models = client.get_available_models(model_type=self._llm_model_type)
        except Exception:
            raise CloudExecutionError("Cirrascale model discovery failed.") from None

        if not isinstance(models, (list, tuple)) or any(
            not isinstance(model, str) or not model.strip() for model in models
        ):
            raise CloudExecutionError("Cirrascale returned an invalid model catalog.")

        self._available_models = tuple(sorted(set(models)))
        return self._available_models

    def execute(self, prompt: str, decision: RouteDecision) -> str:
        return self.execute_observed(prompt, decision).response

    def execute_observed(
        self, prompt: str, decision: RouteDecision
    ) -> ExecutionObservation:
        if decision.selected_device != Device.CLOUD:
            raise CloudExecutionError("Cirrascale requires a cloud routing decision.")
        if decision.analysis.sensitive:
            raise CloudExecutionError("Privacy policy blocked cloud execution.")

        models = self.list_models()
        if decision.model_id not in models:
            available = ", ".join(models) if models else "none"
            raise CloudConfigurationError(
                f"configured cloud model is unavailable; available LLMs: {available}"
            )

        client = self._ensure_client()
        message_factory = self._message_factory
        if message_factory is None:
            raise CloudConfigurationError("Cirrascale message support could not initialize.")

        try:
            started_ns = self._clock_ns()
            response = client.chat(
                messages=[message_factory(role="user", content=prompt)],
                model=decision.model_id,
                max_tokens=decision.analysis.estimated_output_tokens,
                temperature=0,
            )
            finished_ns = self._clock_ns()
            content = response.first_content
        except Exception:
            raise CloudExecutionError("Cirrascale inference failed.") from None

        if not isinstance(content, str) or not content.strip():
            raise CloudExecutionError("Cirrascale returned an empty response.")

        usage = getattr(response, "usage", None)
        prompt_tokens = _optional_nonnegative_int(getattr(usage, "prompt_tokens", None))
        completion_tokens = _optional_nonnegative_int(
            getattr(usage, "completion_tokens", None)
        )
        total_tokens = _optional_nonnegative_int(getattr(usage, "total_tokens", None))
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        response_model = getattr(response, "model", None)
        model_id = response_model if isinstance(response_model, str) and response_model else decision.model_id
        latency_ms = (finished_ns - started_ns) / 1_000_000
        measured_energy_joules = CLOUD_AI_100_TDP_WATTS * (latency_ms / 1000.0)
        return ExecutionObservation(
            response=content,
            api_turnaround_latency_ms=latency_ms,
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            measured_energy_joules=measured_energy_joules,
        )

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = self._environ.get(_API_KEY_ENV)
        endpoint = self._environ.get(_ENDPOINT_ENV)
        if not isinstance(api_key, str) or not api_key.strip():
            raise CloudConfigurationError(f"missing required environment variable {_API_KEY_ENV}.")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise CloudConfigurationError(
                f"missing required environment variable {_ENDPOINT_ENV}."
            )

        endpoint = endpoint.strip().rstrip("/")
        parsed = urlparse(endpoint)
        if (
            parsed.scheme.lower() != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise CloudConfigurationError("cloud endpoint must be a credential-free HTTPS URL.")

        bindings = _load_imagine_bindings()
        try:
            self._client = bindings.client_factory(
                endpoint=endpoint,
                api_key=api_key.strip(),
                max_retries=self._max_retries,
                timeout=self._timeout_seconds,
                verify=True,
                debug=False,
            )
        except Exception:
            raise CloudConfigurationError("Cirrascale cloud support could not initialize.") from None
        self._message_factory = bindings.message_factory
        self._llm_model_type = bindings.llm_model_type
        return self._client


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _default_http_post(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float,
    headers: Mapping[str, str] | None = None,
) -> dict:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read())


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _default_http_get(url: str, *, timeout_seconds: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        return json.loads(response.read())


def phone_health(
    *,
    endpoint: str | None = None,
    environ: Mapping[str, str] | None = None,
    timeout_seconds: float = 10,
    http_get: Callable[[str], dict] | None = None,
) -> dict:
    """Query the phone server's unauthenticated ``/health`` endpoint."""

    environ = environ if environ is not None else os.environ
    resolved_endpoint = (endpoint or environ.get(_PHONE_ENDPOINT_ENV) or "").rstrip("/")
    if not resolved_endpoint:
        raise PhoneConfigurationError(f"missing required environment variable {_PHONE_ENDPOINT_ENV}.")

    get = http_get or (lambda url: _default_http_get(url, timeout_seconds=timeout_seconds))
    try:
        return get(f"{resolved_endpoint}/health")
    except urllib.error.URLError as error:
        raise PhoneConfigurationError(
            f"could not reach the phone server at {resolved_endpoint}: {error.reason}"
        ) from None
    except (TimeoutError, OSError) as error:
        raise PhoneConfigurationError(f"phone server request failed: {error}") from None


class XEliteExecutor:
    """Invoke the local Snapdragon X-Elite Hexagon NPU inference server for PC execution.

    Talks to the OpenAI-compatible ``/v1/chat/completions`` endpoint exposed by
    ``x_elite_laptop_server/serve_qwen_vl.py`` running on this machine.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        environ: Mapping[str, str] | None = None,
        timeout_seconds: float = 120,
        clock_ns: Callable[[], int] = perf_counter_ns,
        http_post: Callable[[str, Mapping[str, Any]], dict] | None = None,
    ) -> None:
        environ = environ if environ is not None else os.environ
        self._endpoint = (
            endpoint or environ.get(_XELITE_ENDPOINT_ENV) or _DEFAULT_XELITE_ENDPOINT
        ).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._clock_ns = clock_ns
        self._http_post = http_post or self._live_http_post

    def _live_http_post(self, url: str, payload: Mapping[str, Any]) -> dict:
        return _default_http_post(url, payload, timeout_seconds=self._timeout_seconds)

    def execute(self, prompt: str, decision: RouteDecision) -> str:
        return self.execute_observed(prompt, decision).response

    def execute_observed(self, prompt: str, decision: RouteDecision) -> ExecutionObservation:
        if decision.selected_device != Device.PC:
            raise PcExecutionError("X-Elite executor requires a PC routing decision.")

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": decision.analysis.estimated_output_tokens,
            "stream": False,
        }

        try:
            started_ns = self._clock_ns()
            body = self._http_post(f"{self._endpoint}/v1/chat/completions", payload)
            finished_ns = self._clock_ns()
        except urllib.error.URLError as error:
            raise PcConfigurationError(
                f"could not reach the local X-Elite server at {self._endpoint}: {error.reason}"
            ) from None
        except (TimeoutError, OSError) as error:
            raise PcConfigurationError(f"local X-Elite server request failed: {error}") from None

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise PcExecutionError("X-Elite server returned an unexpected response shape.") from None
        if not isinstance(content, str) or not content.strip():
            raise PcExecutionError("X-Elite server returned an empty response.")

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        response_model = body.get("model")
        model_id = response_model if isinstance(response_model, str) and response_model else decision.model_id

        profile = body.get("quad_profile") if isinstance(body.get("quad_profile"), dict) else {}
        measured_energy_joules = None
        energy_mj = _optional_float(profile.get("measured_energy_mj"))
        if profile.get("energy_available") is True and energy_mj is not None:
            measured_energy_joules = energy_mj / 1000.0
        compute_unit = profile.get("device")
        backend = profile.get("backend")

        return ExecutionObservation(
            response=content,
            api_turnaround_latency_ms=(finished_ns - started_ns) / 1_000_000,
            model_id=model_id,
            prompt_tokens=_optional_nonnegative_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_nonnegative_int(usage.get("completion_tokens")),
            total_tokens=_optional_nonnegative_int(usage.get("total_tokens")),
            ttft_ms=_optional_float(profile.get("ttft_ms")),
            prefill_speed_tokens_per_second=_optional_float(profile.get("prefill_speed_tok_s")),
            decode_speed_tokens_per_second=_optional_float(profile.get("decode_speed_tok_s")),
            measured_energy_joules=measured_energy_joules,
            tokens_per_joule=_optional_float(profile.get("tokens_per_joule")),
            compute_unit=compute_unit if isinstance(compute_unit, str) else None,
            backend=backend if isinstance(backend, str) else None,
        )


class GenieXPhoneExecutor:
    """Invoke the phone's in-app GenieX inference server for phone execution.

    Talks to the OpenAI-compatible ``/v1/chat/completions`` endpoint exposed
    by the Android app's ``InferenceService`` (see ``s25_android_app``) over
    the LAN -- wireless, not USB, so the phone stays unplugged and its
    on-device energy measurements (see the app's ``measuredNpuPowerMw``)
    stay valid. Requires a bearer token issued by the app (shown in its
    server toggle) since nothing else protects the port.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        token: str | None = None,
        environ: Mapping[str, str] | None = None,
        timeout_seconds: float = 120,
        clock_ns: Callable[[], int] = perf_counter_ns,
        http_post: Callable[[str, Mapping[str, Any]], dict] | None = None,
    ) -> None:
        environ = environ if environ is not None else os.environ
        self._endpoint = (endpoint or environ.get(_PHONE_ENDPOINT_ENV) or "").rstrip("/")
        self._token = token or environ.get(_PHONE_TOKEN_ENV) or ""
        self._timeout_seconds = timeout_seconds
        self._clock_ns = clock_ns
        self._http_post = http_post or self._live_http_post

    def _live_http_post(self, url: str, payload: Mapping[str, Any]) -> dict:
        return _default_http_post(
            url,
            payload,
            timeout_seconds=self._timeout_seconds,
            headers={"Authorization": f"Bearer {self._token}"},
        )

    def execute(self, prompt: str, decision: RouteDecision) -> str:
        return self.execute_observed(prompt, decision).response

    def execute_observed(self, prompt: str, decision: RouteDecision) -> ExecutionObservation:
        if decision.selected_device != Device.PHONE:
            raise PhoneExecutionError("phone executor requires a phone routing decision.")
        if not self._endpoint:
            raise PhoneConfigurationError(f"missing required environment variable {_PHONE_ENDPOINT_ENV}.")
        if not self._token:
            raise PhoneConfigurationError(f"missing required environment variable {_PHONE_TOKEN_ENV}.")

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": decision.analysis.estimated_output_tokens,
            "stream": False,
        }

        try:
            started_ns = self._clock_ns()
            body = self._http_post(f"{self._endpoint}/v1/chat/completions", payload)
            finished_ns = self._clock_ns()
        except urllib.error.URLError as error:
            raise PhoneConfigurationError(
                f"could not reach the phone server at {self._endpoint}: {error.reason}"
            ) from None
        except (TimeoutError, OSError) as error:
            raise PhoneConfigurationError(f"phone server request failed: {error}") from None

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise PhoneExecutionError("phone server returned an unexpected response shape.") from None
        if not isinstance(content, str) or not content.strip():
            raise PhoneExecutionError("phone server returned an empty response.")

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        response_model = body.get("model")
        model_id = response_model if isinstance(response_model, str) and response_model else decision.model_id

        profile = body.get("phone_profile") if isinstance(body.get("phone_profile"), dict) else {}
        measured_energy_joules = None
        energy_mj = _optional_float(profile.get("measured_energy_mj"))
        if profile.get("energy_available") is True and energy_mj is not None:
            measured_energy_joules = energy_mj / 1000.0
        compute_unit = profile.get("compute_unit")

        return ExecutionObservation(
            response=content,
            api_turnaround_latency_ms=(finished_ns - started_ns) / 1_000_000,
            model_id=model_id,
            prompt_tokens=_optional_nonnegative_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_nonnegative_int(usage.get("completion_tokens")),
            total_tokens=_optional_nonnegative_int(usage.get("total_tokens")),
            ttft_ms=_optional_float(profile.get("ttft_ms")),
            prefill_speed_tokens_per_second=_optional_float(profile.get("prefill_speed_tok_s")),
            decode_speed_tokens_per_second=_optional_float(profile.get("decode_speed_tok_s")),
            measured_energy_joules=measured_energy_joules,
            tokens_per_joule=_optional_float(profile.get("tokens_per_joule")),
            compute_unit=compute_unit if isinstance(compute_unit, str) else None,
        )


def default_simulated_executors() -> dict[Device, SimulatedExecutor]:
    return {device: SimulatedExecutor(device) for device in Device}


def build_executors(
    *,
    live_phone: bool = False,
    live_pc: bool = False,
    live_cloud: bool = False,
    phone_executor: GenieXPhoneExecutor | None = None,
    pc_executor: XEliteExecutor | None = None,
    cloud_executor: CirrascaleExecutor | None = None,
) -> dict[Device, Executor]:
    """Combine simulated execution with any opt-in live executors.

    Destinations without a ``live_*`` flag set stay simulated -- this is the
    single entry point behind ``cirrascale_executors``/``x_elite_executors``/
    ``hybrid_executors`` and the CLI's ``--live-phone``/``--live-pc``/
    ``--live-cloud`` flags.
    """

    executors: dict[Device, Executor] = dict(default_simulated_executors())
    if live_phone:
        executors[Device.PHONE] = phone_executor or GenieXPhoneExecutor()
    if live_pc:
        executors[Device.PC] = pc_executor or XEliteExecutor()
    if live_cloud:
        executors[Device.CLOUD] = cloud_executor or CirrascaleExecutor()
    return executors


def cirrascale_executors(
    cloud_executor: CirrascaleExecutor | None = None,
) -> dict[Device, Executor]:
    """Combine simulated local execution with an opt-in live cloud executor."""

    return build_executors(live_cloud=True, cloud_executor=cloud_executor)


def x_elite_executors(
    pc_executor: XEliteExecutor | None = None,
) -> dict[Device, Executor]:
    """Combine simulated phone/cloud execution with a live local X-Elite PC executor."""

    return build_executors(live_pc=True, pc_executor=pc_executor)


def hybrid_executors(
    *,
    cloud_executor: CirrascaleExecutor | None = None,
    pc_executor: XEliteExecutor | None = None,
) -> dict[Device, Executor]:
    """Combine a live cloud executor and a live local X-Elite PC executor; phone stays simulated."""

    return build_executors(live_cloud=True, live_pc=True, cloud_executor=cloud_executor, pc_executor=pc_executor)
