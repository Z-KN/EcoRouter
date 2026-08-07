"""Live smoke test: phone-health plus forced-telemetry --live-phone/--live-pc/--live-cloud runs.

Each forced-telemetry leg loads a snapshot from examples/telemetry/ that marks every device but
the target unavailable, so the router has no choice but to select it, then invokes the real
executor. A leg is SKIPped (not FAILed) when its required environment variables aren't set --
that means the destination isn't configured on this machine, not that it's broken.

Usage: python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peqrouter import (
    Device,
    PEQRouter,
    ExecutionError,
    OptimizationProfile,
    RouteRequest,
    build_executors,
    phone_health,
)
from peqrouter.scenarios import load_telemetry

PROMPT = "Who are you"
TELEMETRY_DIR = Path(__file__).resolve().parent.parent / "examples" / "telemetry"

_results: list[tuple[str, str]] = []


def record(name: str, status: str, detail: str) -> None:
    _results.append((name, status))
    print(f"[{status}] {name}: {detail}")


def make_router() -> PEQRouter:
    return PEQRouter()


def check_phone_health() -> None:
    if not os.environ.get("PHONE_SERVER_ENDPOINT"):
        record("phone-health", "SKIP", "PHONE_SERVER_ENDPOINT not set")
        return
    try:
        health = phone_health()
    except ExecutionError as error:
        record("phone-health", "FAIL", str(error))
        return
    record("phone-health", "PASS", f"status={health.get('status')} model={health.get('model')}")


def run_forced_leg(
    name: str,
    *,
    expected_device: Device,
    origin: Device,
    telemetry_file: str,
    required_env: tuple[str, ...],
    live_flags: dict[str, bool],
) -> None:
    missing = [var for var in required_env if not os.environ.get(var)]
    if missing:
        record(name, "SKIP", f"missing {', '.join(missing)}")
        return

    request = RouteRequest(
        prompt=PROMPT,
        origin=origin,
        telemetry=load_telemetry(TELEMETRY_DIR / telemetry_file),
        profile=OptimizationProfile.BALANCED,
    )
    try:
        result = make_router().run(request, build_executors(**live_flags))
    except ExecutionError as error:
        record(name, "FAIL", str(error))
        return

    if result.decision.selected_device != expected_device:
        record(
            name,
            "FAIL",
            f"router selected {result.decision.selected_device.value}, expected {expected_device.value}",
        )
        return

    detail = f"{result.decision.model_id}: {result.response[:60]!r}"
    metrics = result.metrics
    if metrics is not None and metrics.api_turnaround_latency_ms is not None:
        detail += f" ({metrics.api_turnaround_latency_ms:.0f} ms)"
    record(name, "PASS", detail)


def main() -> int:
    check_phone_health()
    run_forced_leg(
        "live-phone",
        expected_device=Device.PHONE,
        origin=Device.PHONE,
        telemetry_file="force_phone.json",
        required_env=("PHONE_SERVER_ENDPOINT", "PHONE_SERVER_TOKEN"),
        live_flags={"live_phone": True},
    )
    run_forced_leg(
        "live-pc",
        expected_device=Device.PC,
        origin=Device.PC,
        telemetry_file="force_pc.json",
        required_env=(),
        live_flags={"live_pc": True},
    )
    run_forced_leg(
        "live-cloud",
        expected_device=Device.CLOUD,
        origin=Device.PC,
        telemetry_file="force_cloud.json",
        required_env=("INFERENCE_CLOUD_API_KEY", "INFERENCE_CLOUD_ENDPOINT"),
        live_flags={"live_cloud": True},
    )

    passed = sum(1 for _, status in _results if status == "PASS")
    failed = sum(1 for _, status in _results if status == "FAIL")
    skipped = sum(1 for _, status in _results if status == "SKIP")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
