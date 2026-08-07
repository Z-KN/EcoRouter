"""EcoRouter policy engine."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Mapping

from .analyzer import HeuristicPromptAnalyzer, PromptAnalyzer
from .executors import Executor, ObservedExecutor
from .models import (
    CLOUD_AI_100_TDP_WATTS,
    CandidateEvaluation,
    Device,
    DeviceConfig,
    ExecutionError,
    ExecutionMetrics,
    ExecutionResult,
    NoRouteError,
    OptimizationProfile,
    RouteDecision,
    RouteRequest,
    ValidationError,
    default_device_configs,
)

if TYPE_CHECKING:  # imported lazily: the estimator pulls in torch, the router must not
    from .estimator import CalibratedEstimator


_PROFILE_WEIGHTS: dict[OptimizationProfile, dict[str, float]] = {
    OptimizationProfile.BALANCED: {
        "latency": 0.500,
        "energy": 0.333,
        "quality": 0.167,
    },
    OptimizationProfile.LOW_LATENCY: {
        "latency": 0.850,
        "energy": 0.075,
        "quality": 0.075,
    },
    OptimizationProfile.LOW_ENERGY: {
        "latency": 0.150,
        "energy": 0.800,
        "quality": 0.050,
    },
    OptimizationProfile.HIGH_QUALITY: {
        "latency": 0.100,
        "energy": 0.100,
        "quality": 0.800,
    },
}

# Predicted energy, in joules, that maps to a fully-saturated (1.0) energy
# penalty. Chosen against the real per-device constants in scenarios.py
# (calibrated phone/PC J/token, CLOUD_AI_100_TDP_WATTS x latency for cloud),
# not the illustrative dummies those replaced -- those old numbers made every
# real request cost single-digit joules, so the old ceiling of 50 was already
# most of the way to saturated for a typical exchange. Against the real
# numbers, ~9s of sustained 75W cloud generation (a long response) lands
# close to but under this ceiling, leaving room for shorter/cheaper requests
# to differentiate below it instead of everything clamping to 1.0.
_ENERGY_PENALTY_CEILING_JOULES = 700.0

_DEVICE_ORDER = {Device.PHONE: 0, Device.PC: 1, Device.CLOUD: 2}

_UNCALIBRATED_DEVICE_LABELS = {
    Device.PHONE: "phone",
    Device.PC: "PC (X-Elite NPU)",
    Device.CLOUD: "cloud",
}

_MEASURED_ENERGY_SCOPE = {
    Device.PHONE: (
        "measured whole-device battery discharge during decode, NPU-only, "
        "per-model calibration; excludes Wi-Fi radio energy for the request "
        "itself; only valid while the phone is unplugged"
    ),
    Device.PC: (
        "measured whole-laptop battery discharge during sustained NPU-serving "
        "load minus an idle baseline, per-model calibration; covers prefill + "
        "decode; excludes display/background baseline drift; only valid while "
        "the laptop is unplugged"
    ),
    Device.CLOUD: (
        "Qualcomm Cloud AI 100 rated TDP (75 W) x wall-clock request latency; "
        "not an on-device power measurement -- includes network and queueing "
        "time and does not account for multi-tenant sharing of the accelerator"
    ),
}
_DEFAULT_MEASURED_ENERGY_SCOPE = "measured whole-device power during inference minus an idle baseline"

_MEASURED_ENERGY_METHOD = {
    Device.PHONE: "measured_npu_power_x_latency_minus_idle_baseline",
    Device.PC: "measured_npu_power_x_latency_minus_idle_baseline",
    Device.CLOUD: "cloud_accelerator_tdp_x_latency",
}
_DEFAULT_MEASURED_ENERGY_METHOD = "measured_npu_power_x_latency_minus_idle_baseline"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class EcoRouter:
    """Choose one predeployed device/model for a request."""

    def __init__(
        self,
        device_configs: Mapping[Device, DeviceConfig] | None = None,
        analyzer: PromptAnalyzer | None = None,
        estimator: "CalibratedEstimator | None" = None,
    ) -> None:
        self.device_configs = dict(device_configs or default_device_configs())
        if set(self.device_configs) != set(Device):
            raise ValidationError("device configuration must describe phone, pc, and cloud")
        self.analyzer = analyzer if analyzer is not None else HeuristicPromptAnalyzer()
        # Optional. Without it the router keeps its original static behaviour:
        # quality judged by a per-device capability_score and output length
        # guessed by the analyzer. With it, both become per-prompt predictions
        # measured on this hardware. Nothing else about routing changes.
        self.estimator = estimator

    def route(self, request: RouteRequest) -> RouteDecision:
        analysis = self.analyzer.analyze(request.prompt)
        estimate = None
        if self.estimator is not None:
            from .estimator import EstimatorUnavailableError

            try:
                estimate = self.estimator.estimate(request.prompt, intent=analysis.intent.value)
            except EstimatorUnavailableError:
                # Same fallback as never having configured an estimator at
                # all -- a runtime failure (weights no longer cached, torch
                # broken) must not take routing down with it.
                estimate = None
        candidates = tuple(self._evaluate(device, request, analysis, estimate) for device in Device)
        eligible = [item for item in candidates if item.eligible]
        if not eligible:
            reasons = "; ".join(
                f"{item.device.value}: {', '.join(item.exclusion_reasons)}" for item in candidates
            )
            raise NoRouteError("no eligible destination; " + reasons)

        # An untrusted estimate means the prompt is outside what the heads
        # were calibrated on -- the k-NN quality head still returns a
        # confident-looking number, but it is not evidence. Rather than let
        # that number (via the static-capability fallback) get outscored by a
        # cheap device on latency/energy, default straight to the
        # highest-capability destination: cloud, or PC if the prompt cannot
        # leave the device at all. If that destination is not itself
        # eligible (offline, etc.) fall through to the existing
        # highest-capability-eligible logic below instead of erroring.
        untrusted_fallback = None
        if estimate is not None and not estimate.trusted:
            fallback_device = Device.PC if analysis.sensitive else Device.CLOUD
            untrusted_fallback = next(
                (item for item in eligible if item.device == fallback_device), None
            )

        if untrusted_fallback is not None:
            selected = untrusted_fallback
            quality_degraded = True
            if analysis.sensitive:
                explanation = (
                    f"Selected {selected.device.value}/{selected.model_id}: the quality estimate was "
                    "out of the calibration domain (untrusted) and the prompt is privacy-sensitive, "
                    "so routing defaulted to PC as the highest-capability privacy-safe device rather "
                    "than trust an extrapolated per-device prediction."
                )
            else:
                explanation = (
                    f"Selected {selected.device.value}/{selected.model_id}: the quality estimate was "
                    "out of the calibration domain (untrusted), so routing defaulted to cloud, the "
                    "highest-capability device, rather than trust an extrapolated per-device prediction."
                )
        else:
            sufficient = [item for item in eligible if item.quality_sufficient]
            quality_degraded = not sufficient
            if sufficient:
                selected = min(sufficient, key=self._score_sort_key)
                explanation = (
                    f"Selected {selected.device.value}/{selected.model_id}: it had the lowest "
                    f"{request.profile.value} score among privacy-safe, healthy models meeting required quality."
                )
            else:
                selected = min(
                    eligible,
                    key=lambda item: (
                        -self.device_configs[item.device].capability_score,
                        *self._score_sort_key(item),
                    ),
                )
                explanation = (
                    f"Selected {selected.device.value}/{selected.model_id} as the highest-capability eligible "
                    "model; no privacy-safe, healthy destination met the requested quality."
                )

        # Head B's p90 replaces the analyzer's guess in the reported decision
        # (SimulatedExecutor's receipt, the "Est. output tokens" stat,
        # ExecutionResult.to_dict()). It is not sent to live executors as a
        # generation cap -- see _MAX_GENERATION_TOKENS in executors.py, which
        # exists precisely because a *predicted* length must never truncate a
        # real answer that runs longer than predicted.
        if estimate is not None and estimate.trusted:
            capped = estimate.length_p90.get(selected.device)
            if capped:
                analysis = replace(analysis, estimated_output_tokens=capped)

        return RouteDecision(
            selected_device=selected.device,
            model_id=selected.model_id,
            profile=request.profile,
            analysis=analysis,
            quality_degraded=quality_degraded,
            explanation=explanation,
            candidates=candidates,
        )

    def run(self, request: RouteRequest, executors: Mapping[Device, Executor]) -> ExecutionResult:
        decision = self.route(request)
        executor = executors.get(decision.selected_device)
        if executor is None:
            raise ExecutionError(f"no executor registered for {decision.selected_device.value}")
        metrics = None
        if isinstance(executor, ObservedExecutor):
            observation = executor.execute_observed(request.prompt, decision)
            telemetry = request.telemetry[decision.selected_device]
            estimated_energy = (
                observation.total_tokens * telemetry.energy_joules_per_token
                if observation.total_tokens is not None
                else None
            )
            if observation.measured_energy_joules is not None:
                energy_estimate_method = _MEASURED_ENERGY_METHOD.get(
                    decision.selected_device, _DEFAULT_MEASURED_ENERGY_METHOD
                )
                energy_scope = _MEASURED_ENERGY_SCOPE.get(
                    decision.selected_device, _DEFAULT_MEASURED_ENERGY_SCOPE
                )
                confidence = "measured"
            else:
                device_label = _UNCALIBRATED_DEVICE_LABELS[decision.selected_device]
                energy_estimate_method = "actual_total_tokens_x_configured_joules_per_token"
                energy_scope = f"uncalibrated {device_label} inference estimate"
                confidence = "uncalibrated"
            metrics = ExecutionMetrics(
                api_turnaround_latency_ms=observation.api_turnaround_latency_ms,
                prompt_tokens=observation.prompt_tokens,
                completion_tokens=observation.completion_tokens,
                total_tokens=observation.total_tokens,
                measured_energy_joules=observation.measured_energy_joules,
                estimated_energy_joules=estimated_energy,
                energy_joules_per_token=telemetry.energy_joules_per_token,
                energy_estimate_method=energy_estimate_method,
                energy_scope=energy_scope,
                confidence=confidence,
                ttft_ms=observation.ttft_ms,
                prefill_speed_tokens_per_second=observation.prefill_speed_tokens_per_second,
                decode_speed_tokens_per_second=observation.decode_speed_tokens_per_second,
                tokens_per_joule=observation.tokens_per_joule,
                compute_unit=observation.compute_unit,
                backend=observation.backend,
            )
            response = observation.response
        else:
            response = executor.execute(request.prompt, decision)
        return ExecutionResult(decision=decision, response=response, metrics=metrics)

    def _evaluate(
        self, device: Device, request: RouteRequest, analysis, estimate=None
    ) -> CandidateEvaluation:
        telemetry = request.telemetry[device]
        config = self.device_configs[device]
        exclusions: list[str] = []
        if not telemetry.available:
            exclusions.append("unavailable")
        if device == Device.CLOUD and analysis.sensitive:
            exclusions.append("cloud blocked by privacy policy")

        # Head B, when trusted, replaces the analyzer's guess at how long the
        # answer runs. This is the input both the latency and the energy
        # formula divide and multiply by, so it is the single number that most
        # changes what routing picks -- and the one most worth measuring.
        output_tokens = analysis.estimated_output_tokens
        if estimate is not None and estimate.length_p50.get(device):
            output_tokens = estimate.length_p50[device]

        total_tokens = analysis.estimated_input_tokens + output_tokens
        can_estimate = telemetry.throughput_tokens_per_second > 0
        network_latency = 0.0 if device == request.origin else telemetry.network_latency_ms
        latency = (
            network_latency + total_tokens / telemetry.throughput_tokens_per_second * 1000
            if can_estimate
            else None
        )
        if not can_estimate:
            energy = None
        elif device == Device.CLOUD:
            # Cloud has no meaningful per-token energy constant of its own --
            # it's a shared accelerator running at a roughly fixed draw for
            # however long the request takes, not proportional token-by-token
            # work the way a dedicated local NPU is. Predicted energy is
            # therefore rated TDP x predicted latency (the same formula
            # CirrascaleExecutor uses for *measured* energy after the call),
            # not tokens x a synthetic J/token rate.
            energy = CLOUD_AI_100_TDP_WATTS * (latency / 1000.0)
        else:
            energy = total_tokens * telemetry.energy_joules_per_token
        cloud_cost = total_tokens / 1000 * telemetry.cloud_cost_per_1k_tokens_usd if can_estimate else None

        penalties: dict[str, float] = {}
        score = None
        if can_estimate:
            penalties = {
                "latency": _clamp(latency / 10_000),
                "energy": _clamp(energy / _ENERGY_PENALTY_CEILING_JOULES),
                # Static capability_score, always scored. The calibrated
                # per-prompt p_pass (when trusted) already decides *eligibility*
                # in _quality_sufficient below and saturates at 1.0 for both PC
                # and cloud on most prompts -- it cannot tell a profile that
                # wants the more capable device apart from one that is
                # indifferent between two devices that both merely pass. This
                # is a coarser, always-available preference for that: it never
                # readmits a device that failed the hard gate, it only ranks
                # among devices that already passed it.
                "quality": 1.0 - config.capability_score,
            }
            weights = _PROFILE_WEIGHTS[request.profile]
            applicable_weight = sum(weights[name] for name in penalties)
            score = sum(weights[name] * value for name, value in penalties.items()) / applicable_weight

        return CandidateEvaluation(
            device=device,
            model_id=config.model_id,
            eligible=not exclusions,
            exclusion_reasons=tuple(exclusions),
            quality_sufficient=self._quality_sufficient(device, config, analysis, estimate),
            predicted_latency_ms=latency,
            predicted_energy_joules=energy,
            predicted_cloud_cost_usd=cloud_cost,
            score=score,
            penalties=penalties,
        )

    def _quality_sufficient(self, device: Device, config, analysis, estimate) -> bool:
        """Decide the quality gate for one device.

        Falls back to the static capability comparison in two cases: no
        estimator is configured, or the prompt sits outside the calibration
        domain. The second matters more than it looks -- a k-NN head always
        returns a number, and for a prompt unlike anything measured that number
        is confident and meaningless. Abstaining back to the documented default
        is the honest failure mode.
        """

        if estimate is None or not estimate.trusted:
            return config.capability_score >= analysis.required_quality
        predicted = estimate.p_pass.get(device)
        if predicted is None:
            return config.capability_score >= analysis.required_quality
        return predicted >= self.estimator.quality_floor

    @staticmethod
    def _score_sort_key(item: CandidateEvaluation) -> tuple[float, float, int]:
        return (
            item.score if item.score is not None else float("inf"),
            item.predicted_latency_ms if item.predicted_latency_ms is not None else float("inf"),
            _DEVICE_ORDER[item.device],
        )
