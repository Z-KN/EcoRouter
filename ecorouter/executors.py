"""Execution adapter contract and dependency-free simulators."""

from __future__ import annotations

from typing import Protocol

from .models import Device, RouteDecision


class Executor(Protocol):
    def execute(self, prompt: str, decision: RouteDecision) -> str:
        """Execute a prompt on the selected device and return its response."""


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


def default_simulated_executors() -> dict[Device, SimulatedExecutor]:
    return {device: SimulatedExecutor(device) for device in Device}
