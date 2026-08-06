"""EcoRouter policy engine."""

from __future__ import annotations

from typing import Mapping

from .analyzer import PresidioPromptAnalyzer, PromptAnalyzer
from .executors import (
    Executor,
    ObservedExecutor,
    cirrascale_executors,
    default_simulated_executors,
    hybrid_executors,
    x_elite_executors,
)
from .models import (
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


_PROFILE_WEIGHTS: dict[OptimizationProfile, dict[str, float]] = {
    OptimizationProfile.BALANCED: {
        "latency": 0.30,
        "energy": 0.20,
        "utilization": 0.10,
        "thermal": 0.10,
        "battery": 0.10,
        "cost": 0.10,
        "quality": 0.10,
    },
    OptimizationProfile.LOW_LATENCY: {
        "latency": 0.55,
        "energy": 0.10,
        "utilization": 0.10,
        "thermal": 0.05,
        "battery": 0.05,
        "cost": 0.05,
        "quality": 0.10,
    },
    OptimizationProfile.ENERGY_SAVER: {
        "latency": 0.15,
        "energy": 0.40,
        "utilization": 0.10,
        "thermal": 0.15,
        "battery": 0.10,
        "cost": 0.05,
        "quality": 0.05,
    },
    OptimizationProfile.HIGH_QUALITY: {
        "latency": 0.15,
        "energy": 0.10,
        "utilization": 0.05,
        "thermal": 0.05,
        "battery": 0.05,
        "cost": 0.10,
        "quality": 0.50,
    },
}

_DEVICE_ORDER = {Device.PHONE: 0, Device.PC: 1, Device.CLOUD: 2}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class EcoRouter:
    """Choose one predeployed device/model for a request."""

    def __init__(
        self,
        device_configs: Mapping[Device, DeviceConfig] | None = None,
        analyzer: PromptAnalyzer | None = None,
    ) -> None:
        self.device_configs = dict(device_configs or default_device_configs())
        if set(self.device_configs) != set(Device):
            raise ValidationError("device configuration must describe phone, pc, and cloud")
        self.analyzer = analyzer if analyzer is not None else PresidioPromptAnalyzer()

    def route(self, request: RouteRequest) -> RouteDecision:
        analysis = self.analyzer.analyze(request.prompt)
        candidates = tuple(self._evaluate(device, request, analysis) for device in Device)
        eligible = [item for item in candidates if item.eligible]
        if not eligible:
            reasons = "; ".join(
                f"{item.device.value}: {', '.join(item.exclusion_reasons)}" for item in candidates
            )
            raise NoRouteError("no eligible destination; " + reasons)

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
            metrics = ExecutionMetrics(
                api_turnaround_latency_ms=observation.api_turnaround_latency_ms,
                prompt_tokens=observation.prompt_tokens,
                completion_tokens=observation.completion_tokens,
                total_tokens=observation.total_tokens,
                measured_energy_joules=None,
                estimated_energy_joules=estimated_energy,
                energy_joules_per_token=telemetry.energy_joules_per_token,
                energy_estimate_method="actual_total_tokens_x_configured_joules_per_token",
                energy_scope="uncalibrated cloud inference estimate",
                confidence="uncalibrated",
            )
            response = observation.response
        else:
            response = executor.execute(request.prompt, decision)
        return ExecutionResult(decision=decision, response=response, metrics=metrics)

    def _evaluate(self, device: Device, request: RouteRequest, analysis) -> CandidateEvaluation:
        telemetry = request.telemetry[device]
        config = self.device_configs[device]
        exclusions: list[str] = []
        if not telemetry.available:
            exclusions.append("unavailable")
        if device != Device.CLOUD and telemetry.battery_percent is not None and telemetry.battery_percent <= 5:
            exclusions.append("battery at or below 5%")
        if device != Device.CLOUD and telemetry.thermal_pressure >= 0.95:
            exclusions.append("thermal pressure at or above 0.95")
        if device == Device.CLOUD and analysis.sensitive:
            exclusions.append("cloud blocked by privacy policy")

        total_tokens = analysis.estimated_input_tokens + analysis.estimated_output_tokens
        can_estimate = telemetry.throughput_tokens_per_second > 0
        network_latency = 0.0 if device == request.origin else telemetry.network_latency_ms
        latency = (
            network_latency + total_tokens / telemetry.throughput_tokens_per_second * 1000
            if can_estimate
            else None
        )
        energy = total_tokens * telemetry.energy_joules_per_token if can_estimate else None
        cloud_cost = total_tokens / 1000 * telemetry.cloud_cost_per_1k_tokens_usd if can_estimate else None

        penalties: dict[str, float] = {}
        score = None
        if can_estimate:
            penalties = {
                "latency": _clamp(latency / 10_000),
                "energy": _clamp(energy / 50),
                "utilization": telemetry.utilization,
                "thermal": telemetry.thermal_pressure,
                "cost": _clamp(cloud_cost / 0.10),
                "quality": 1.0 - config.capability_score,
            }
            if telemetry.battery_percent is not None:
                penalties["battery"] = 1.0 - telemetry.battery_percent / 100
            weights = _PROFILE_WEIGHTS[request.profile]
            applicable_weight = sum(weights[name] for name in penalties)
            score = sum(weights[name] * value for name, value in penalties.items()) / applicable_weight

        return CandidateEvaluation(
            device=device,
            model_id=config.model_id,
            eligible=not exclusions,
            exclusion_reasons=tuple(exclusions),
            quality_sufficient=config.capability_score >= analysis.required_quality,
            predicted_latency_ms=latency,
            predicted_energy_joules=energy,
            predicted_cloud_cost_usd=cloud_cost,
            score=score,
            penalties=penalties,
        )

    @staticmethod
    def _score_sort_key(item: CandidateEvaluation) -> tuple[float, float, int]:
        return (
            item.score if item.score is not None else float("inf"),
            item.predicted_latency_ms if item.predicted_latency_ms is not None else float("inf"),
            _DEVICE_ORDER[item.device],
        )


def default_executor_map(
    *,
    live_cloud: bool = False,
    live_pc: bool = False,
) -> dict[Device, Executor]:
    """Build the three-device executor map used by router.run()."""

    if live_cloud and live_pc:
        return hybrid_executors()
    if live_cloud:
        return cirrascale_executors()
    if live_pc:
        return x_elite_executors()
    return default_simulated_executors()
