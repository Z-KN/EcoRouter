"""Typed data contracts for EcoRouter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EcoRouterError(Exception):
    """Base class for expected EcoRouter failures."""


class ValidationError(EcoRouterError):
    """Raised when a request or configuration is invalid."""


class NoRouteError(EcoRouterError):
    """Raised when every destination is excluded by hard constraints."""


class ExecutionError(EcoRouterError):
    """Raised when the selected destination has no usable executor."""


class CloudConfigurationError(ExecutionError):
    """Raised when live cloud execution is not safely configured."""


class CloudExecutionError(ExecutionError):
    """Raised when a configured cloud provider cannot complete a request."""


class PcConfigurationError(ExecutionError):
    """Raised when the local NPU inference server is unreachable or misconfigured."""


class PcExecutionError(ExecutionError):
    """Raised when the local NPU inference server cannot complete a request."""


class PhoneConfigurationError(ExecutionError):
    """Raised when the phone's in-app inference server is unreachable or misconfigured."""


class PhoneExecutionError(ExecutionError):
    """Raised when the phone's in-app inference server cannot complete a request."""


class PrivacyError(EcoRouterError):
    """Base class for fail-closed privacy analyzer failures."""


class PrivacyInitializationError(PrivacyError):
    """Raised when Presidio or its NLP model cannot initialize."""


class PrivacyAnalysisError(PrivacyError):
    """Raised when Presidio cannot safely analyze a prompt."""


class Device(str, Enum):
    PHONE = "phone"
    PC = "pc"
    CLOUD = "cloud"


class OptimizationProfile(str, Enum):
    BALANCED = "balanced"
    LOW_LATENCY = "low-latency"
    ENERGY_SAVER = "energy-saver"
    HIGH_QUALITY = "high-quality"


class Intent(str, Enum):
    LOOKUP = "lookup"
    SUMMARIZATION = "summarization"
    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    GENERAL = "general"


def _require_range(name: str, value: float, minimum: float, maximum: float) -> None:
    _require_number(name, value)
    if not minimum <= value <= maximum:
        raise ValidationError(f"{name} must be between {minimum} and {maximum}")


def _require_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number")


@dataclass(frozen=True)
class DeviceTelemetry:
    available: bool
    network_latency_ms: float
    throughput_tokens_per_second: float
    energy_joules_per_token: float
    utilization: float
    thermal_pressure: float
    battery_percent: float | None = None
    cloud_cost_per_1k_tokens_usd: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise ValidationError("available must be true or false")
        _require_number("network_latency_ms", self.network_latency_ms)
        if self.network_latency_ms < 0:
            raise ValidationError("network_latency_ms must be non-negative")
        _require_number("throughput_tokens_per_second", self.throughput_tokens_per_second)
        if self.throughput_tokens_per_second < 0:
            raise ValidationError("throughput_tokens_per_second must be non-negative")
        if self.available and self.throughput_tokens_per_second == 0:
            raise ValidationError("available devices need positive throughput")
        _require_number("energy_joules_per_token", self.energy_joules_per_token)
        if self.energy_joules_per_token < 0:
            raise ValidationError("energy_joules_per_token must be non-negative")
        _require_range("utilization", self.utilization, 0.0, 1.0)
        _require_range("thermal_pressure", self.thermal_pressure, 0.0, 1.0)
        if self.battery_percent is not None:
            _require_range("battery_percent", self.battery_percent, 0.0, 100.0)
        _require_number("cloud_cost_per_1k_tokens_usd", self.cloud_cost_per_1k_tokens_usd)
        if self.cloud_cost_per_1k_tokens_usd < 0:
            raise ValidationError("cloud_cost_per_1k_tokens_usd must be non-negative")


@dataclass(frozen=True)
class DeviceConfig:
    model_id: str
    capability_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str):
            raise ValidationError("model_id must be a string")
        if not self.model_id.strip():
            raise ValidationError("model_id must not be empty")
        _require_range("capability_score", self.capability_score, 0.0, 1.0)


def default_device_configs() -> dict[Device, DeviceConfig]:
    return {
        Device.PHONE: DeviceConfig("phone-model", 0.60),
        Device.PC: DeviceConfig("pc-model", 0.80),
        Device.CLOUD: DeviceConfig("Llama-3.1-8B", 0.95),
    }


@dataclass(frozen=True)
class RouteRequest:
    prompt: str
    origin: Device
    telemetry: Mapping[Device, DeviceTelemetry]
    profile: OptimizationProfile = OptimizationProfile.BALANCED

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise ValidationError("prompt must be a string")
        if not self.prompt.strip():
            raise ValidationError("prompt must not be empty")
        if not isinstance(self.origin, Device):
            raise ValidationError("origin must be a Device value")
        if self.origin not in (Device.PHONE, Device.PC):
            raise ValidationError("origin must be phone or pc")
        if not isinstance(self.profile, OptimizationProfile):
            raise ValidationError("profile must be an OptimizationProfile value")
        missing = set(Device) - set(self.telemetry)
        extra = set(self.telemetry) - set(Device)
        if missing or extra:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(item.value for item in missing)))
            if extra:
                details.append("unknown " + ", ".join(str(item) for item in extra))
            raise ValidationError("telemetry must describe phone, pc, and cloud (" + "; ".join(details) + ")")
        if any(not isinstance(value, DeviceTelemetry) for value in self.telemetry.values()):
            raise ValidationError("telemetry values must be DeviceTelemetry objects")


@dataclass(frozen=True)
class PromptAnalysis:
    intent: Intent
    complexity: float
    sensitive: bool
    pii_categories: tuple[str, ...]
    estimated_input_tokens: int
    estimated_output_tokens: int
    required_quality: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "complexity": round(self.complexity, 4),
            "sensitive": self.sensitive,
            "pii_categories": list(self.pii_categories),
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "required_quality": round(self.required_quality, 4),
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    device: Device
    model_id: str
    eligible: bool
    exclusion_reasons: tuple[str, ...]
    quality_sufficient: bool
    predicted_latency_ms: float | None
    predicted_energy_joules: float | None
    predicted_cloud_cost_usd: float | None
    score: float | None
    penalties: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device.value,
            "model_id": self.model_id,
            "eligible": self.eligible,
            "exclusion_reasons": list(self.exclusion_reasons),
            "quality_sufficient": self.quality_sufficient,
            "predicted_latency_ms": _rounded(self.predicted_latency_ms),
            "predicted_energy_joules": _rounded(self.predicted_energy_joules),
            "predicted_cloud_cost_usd": _rounded(self.predicted_cloud_cost_usd, 6),
            "score": _rounded(self.score, 6),
            "penalties": {key: round(value, 6) for key, value in self.penalties.items()},
        }


def _rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(frozen=True)
class RouteDecision:
    selected_device: Device
    model_id: str
    profile: OptimizationProfile
    analysis: PromptAnalysis
    quality_degraded: bool
    explanation: str
    candidates: tuple[CandidateEvaluation, ...]

    @property
    def selected_candidate(self) -> CandidateEvaluation:
        return next(item for item in self.candidates if item.device == self.selected_device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_device": self.selected_device.value,
            "model_id": self.model_id,
            "profile": self.profile.value,
            "analysis": self.analysis.to_dict(),
            "quality_degraded": self.quality_degraded,
            "explanation": self.explanation,
            "predicted": {
                "latency_ms": _rounded(self.selected_candidate.predicted_latency_ms),
                "energy_joules": _rounded(self.selected_candidate.predicted_energy_joules),
                "cloud_cost_usd": _rounded(self.selected_candidate.predicted_cloud_cost_usd, 6),
            },
            "candidates": [item.to_dict() for item in self.candidates],
        }


@dataclass(frozen=True)
class ExecutionObservation:
    """Provider observations captured around a single executor call.

    ``ttft_ms`` through ``backend`` are optional performance/energy stats a
    provider may expose alongside the response -- populated for the phone
    (measured) and PC (device/backend only; no on-device energy telemetry)
    executors, left ``None`` for cloud.
    """

    response: str
    api_turnaround_latency_ms: float | None = None
    model_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    ttft_ms: float | None = None
    prefill_speed_tokens_per_second: float | None = None
    decode_speed_tokens_per_second: float | None = None
    measured_energy_joules: float | None = None
    tokens_per_joule: float | None = None
    compute_unit: str | None = None
    backend: str | None = None


@dataclass(frozen=True)
class ExecutionMetrics:
    """Optional live measurements and explicitly labeled energy estimates."""

    api_turnaround_latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    measured_energy_joules: float | None
    estimated_energy_joules: float | None
    energy_joules_per_token: float
    energy_estimate_method: str
    energy_scope: str
    confidence: str
    ttft_ms: float | None = None
    prefill_speed_tokens_per_second: float | None = None
    decode_speed_tokens_per_second: float | None = None
    tokens_per_joule: float | None = None
    compute_unit: str | None = None
    backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_turnaround_latency_ms": _rounded(self.api_turnaround_latency_ms, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "measured_energy_joules": _rounded(self.measured_energy_joules, 6),
            "estimated_energy_joules": _rounded(self.estimated_energy_joules, 6),
            "energy_joules_per_token": round(self.energy_joules_per_token, 6),
            "energy_estimate_method": self.energy_estimate_method,
            "energy_scope": self.energy_scope,
            "confidence": self.confidence,
            "ttft_ms": _rounded(self.ttft_ms, 3),
            "prefill_speed_tokens_per_second": _rounded(self.prefill_speed_tokens_per_second),
            "decode_speed_tokens_per_second": _rounded(self.decode_speed_tokens_per_second),
            "tokens_per_joule": _rounded(self.tokens_per_joule),
            "compute_unit": self.compute_unit,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class ExecutionResult:
    decision: RouteDecision
    response: str
    metrics: ExecutionMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"decision": self.decision.to_dict(), "response": self.response}
        if self.metrics is not None:
            result["metrics"] = self.metrics.to_dict()
        return result
