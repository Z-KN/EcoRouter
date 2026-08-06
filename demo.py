"""EcoRouter demo: six routing decisions, one live execution, ~3 minutes.

Routing decisions cost milliseconds; generations cost seconds. So the demo
shows six decisions instantly and then executes exactly one of them for real,
which is what makes the rest credible -- the same code path produced all six.

Each decision prints the evidence behind it: the calibration prompts the
quality estimate came from, the measured energy constant used, and which gate
(if any) removed a device from consideration.

Usage:
    python demo.py                 # decisions only
    python demo.py --live          # also execute one prompt on real hardware
    python demo.py --profile energy-saver
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from ecorouter.analyzer import HeuristicPromptAnalyzer
from ecorouter.estimator import CalibratedEstimator, EstimatorUnavailableError
from ecorouter.executors import build_executors
from ecorouter.models import (
    Device,
    OptimizationProfile,
    RouteRequest,
    default_device_configs,
)
from ecorouter.router import EcoRouter
from ecorouter.scenarios import built_in_scenarios

HEADS_DIR = Path(__file__).parent / "benchmarks" / "calibration" / "heads"

BAR = "=" * 78


def calibrated_telemetry(base, estimator: CalibratedEstimator):
    """Replace placeholder energy/throughput constants with measured ones.

    The shipped telemetry fixtures carry round numbers (phone 0.010 J/token,
    PC 0.025) that predate any measurement. The sweep measured the real ones,
    and they differ by more than an order of magnitude on the PC -- so leaving
    the fixtures in place would have the router rank on energy it never
    observed while printing energy it did.
    """

    calibrated = {}
    for device, telemetry in base.items():
        j_per_token = estimator.energy_joules_per_token(device)
        decode = estimator.decode_tokens_per_second(device)
        updates = {}
        if j_per_token is not None:
            updates["energy_joules_per_token"] = j_per_token
        if decode is not None:
            updates["throughput_tokens_per_second"] = decode
        calibrated[device] = replace(telemetry, **updates) if updates else telemetry
    return calibrated


def preflight(device: Device, *, attempts: int = 3, pause: float = 20.0) -> bool:
    """Check the phone will answer before executing in front of an audience.

    The phone serves one generation at a time and keeps working after a client
    gives up waiting, so a single earlier timeout leaves it returning 429 for
    minutes. Discovering that mid-demo looks like a broken router; discovering
    it here costs one cheap request.
    """

    import json as _json
    import time
    import urllib.request

    endpoint = os.environ.get("PHONE_SERVER_ENDPOINT", "").rstrip("/")
    token = os.environ.get("PHONE_SERVER_TOKEN", "")
    if device is not Device.PHONE or not endpoint or not token:
        return True

    payload = _json.dumps(
        {"messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 8, "stream": False}
    ).encode()
    for attempt in range(attempts):
        request = urllib.request.Request(
            f"{endpoint}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = _json.loads(response.read().decode())
            speed = (body.get("phone_profile") or {}).get("decode_speed_tok_s")
            print(f"  preflight : phone ready" + (f" ({speed:.0f} tok/s)" if speed else ""))
            return True
        except Exception:  # noqa: BLE001 - busy, slow, or unreachable all mean "wait"
            if attempt < attempts - 1:
                print(f"  preflight : phone busy, waiting {pause:.0f}s ...")
                time.sleep(pause)
    return False


def show(router: EcoRouter, estimator, telemetry, prompt: str, profile, note: str) -> None:
    decision = router.route(RouteRequest(prompt, Device.PC, telemetry, profile))
    estimate = (
        estimator.estimate(prompt, intent=router.analyzer.analyze(prompt).intent.value)
        if estimator
        else None
    )
    selected = next(c for c in decision.candidates if c.device == decision.selected_device)

    print(f"\n{BAR}\n{note}\n  prompt : {prompt[:100]}")
    print(f"  profile: {profile.value}")
    print(f"  -> {decision.selected_device.value.upper()} / {decision.model_id}")

    if selected.predicted_latency_ms is not None:
        line = f"     predicted {selected.predicted_latency_ms:.0f} ms"
        if selected.predicted_energy_joules is not None:
            line += f", {selected.predicted_energy_joules:.2f} J"
        if selected.predicted_cloud_cost_usd:
            line += f", ${selected.predicted_cloud_cost_usd:.5f}"
        print(line)

    for candidate in decision.candidates:
        marks = []
        if candidate.exclusion_reasons:
            marks.append("BLOCKED: " + ", ".join(candidate.exclusion_reasons))
        elif not candidate.quality_sufficient:
            marks.append("below quality floor")
        if estimate is not None:
            p = estimate.p_pass.get(candidate.device)
            marks.append(f"P(pass)={p:.2f}" if p is not None else "P(pass) uncalibrated")
            predicted_len = estimate.length_p50.get(candidate.device, 0)
            # 0 means head B had no labelled neighbour for this device, in
            # which case the router used the analyzer's guess. Printing "~0 tok"
            # would read as "this device answers for free".
            marks.append(f"~{predicted_len} tok" if predicted_len else "len from analyzer")
        flag = "*" if candidate.device == decision.selected_device else " "
        print(f"   {flag} {candidate.device.value:<6} {'; '.join(marks)}")

    if estimate is not None:
        if not estimate.trusted:
            print(f"     ! out of calibration domain (d={estimate.mean_distance:.3f})"
                  " - quality gate abstained, ranked on latency/energy/cost only")
        else:
            cited = ", ".join(f"{pid}({sim:.2f})" for pid, _, sim in estimate.neighbours[:3])
            print(f"     evidence: nearest calibration prompts {cited}")
    if decision.quality_degraded:
        print("     ! no destination met the quality floor; picked highest capability")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Execute one prompt on real hardware.")
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--no-estimator", action="store_true", help="Show pre-calibration behaviour.")
    parser.add_argument(
        "--live-prompt",
        default="What is 15 percent of 240?",
        help=(
            "Prompt to execute for real. The default routes to the phone, which "
            "shows measured NPU energy but currently pays a ~90s HTTP round trip "
            "on that server. For a timed demo prefer a prompt that routes off "
            "device, e.g. the multi-step train question, which lands on cloud in "
            "about a second."
        ),
    )
    parser.add_argument(
        "--preflight-attempts",
        type=int,
        default=3,
        help="How many times to check the phone is free before executing live.",
    )
    args = parser.parse_args()

    estimator = None
    if not args.no_estimator:
        try:
            estimator = CalibratedEstimator(HEADS_DIR)
        except EstimatorUnavailableError as error:
            print(f"! running uncalibrated: {error}\n")

    # Name the models that actually answered during calibration. The shipped
    # fixtures still say "phone-model"/"pc-model", which would make the demo
    # claim less than it can prove.
    device_configs = dict(default_device_configs())
    if estimator is not None:
        for device, config in device_configs.items():
            observed = estimator.observed_model_id(device)
            if observed:
                device_configs[device] = replace(config, model_id=observed)

    router = EcoRouter(
        device_configs=device_configs,
        analyzer=HeuristicPromptAnalyzer(),
        estimator=estimator,
    )
    telemetry = built_in_scenarios()["healthy"]
    if estimator is not None:
        telemetry = calibrated_telemetry(telemetry, estimator)
        print("Calibrated device constants (measured, medians over the sweep):")
        for device in Device:
            j = estimator.energy_joules_per_token(device)
            d = estimator.decode_tokens_per_second(device)
            print(
                f"  {device.value:<6} "
                f"{(f'{j:.4f} J/token' if j else 'no energy telemetry'):<22}"
                f"{(f'{d:.1f} tok/s' if d else ''):<14}"
            )

    profile = OptimizationProfile(args.profile)

    show(router, estimator, telemetry,
         "What is 15 percent of 240?",
         profile,
         "[1] Easy arithmetic - the phone is competent, so use the cheapest tier.")

    show(router, estimator, telemetry,
         "A train leaves at 2:15 PM travelling 80 km/h. A second train leaves the same "
         "station at 3:00 PM travelling 100 km/h in the same direction. At what time "
         "does the second train catch the first?",
         profile,
         "[2] Multi-step reasoning - the phone's P(pass) drops below the floor.")

    show(router, estimator, telemetry,
         "Summarize the medical history for John Smith, SSN 123-45-6789.",
         OptimizationProfile.HIGH_QUALITY,
         "[3] PII under HIGH_QUALITY - cloud would win on merit and is hard-blocked anyway.")

    # The flip is driven by telemetry, not by the optimization profile. On this
    # hardware the phone measured fastest *and* most efficient, so it dominates
    # on every weighted axis and no profile can reorder the ranking -- verified
    # across all four profiles and five prompts. Device state can, though: drop
    # the phone's battery and the same prompt leaves the device.
    low_battery = built_in_scenarios()["phone-low-battery"]
    if estimator is not None:
        low_battery = calibrated_telemetry(low_battery, estimator)

    show(router, estimator, telemetry,
         "What is 15 percent of 240?",
         OptimizationProfile.ENERGY_SAVER,
         "[4] Same easy prompt, healthy device - stays on the phone.")

    show(router, estimator, low_battery,
         "What is 15 percent of 240?",
         OptimizationProfile.ENERGY_SAVER,
         "[5] Same prompt, phone battery low - routing leaves the device.")

    show(router, estimator, telemetry,
         "Write a poem about the smell of rain on hot asphalt.",
         profile,
         "[6] Nothing like this was calibrated - the estimator should abstain, not guess.")

    if args.live:
        print(f"\n{BAR}\nLIVE EXECUTION - same code path, real hardware")
        if not preflight(Device.PHONE, attempts=args.preflight_attempts):
            print("  ! phone is busy or slow; run again in a minute, or use --no-estimator")
            return 1
        prompt = args.live_prompt
        request = RouteRequest(prompt, Device.PC, telemetry, profile)
        decision = router.route(request)
        executors = build_executors(live_phone=True, live_pc=True, live_cloud=True)
        result = router.run(request, executors)
        selected = next(c for c in decision.candidates if c.device == decision.selected_device)
        print(f"  routed to : {decision.selected_device.value}")
        print(f"  response  : {result.response.strip()[:160]}")
        if result.metrics:
            m = result.metrics
            print(f"  measured  : {m.api_turnaround_latency_ms:.0f} ms"
                  + (f", {m.measured_energy_joules:.3f} J" if m.measured_energy_joules else "")
                  + (f", {m.completion_tokens} tokens" if m.completion_tokens else ""))
            # Predicted vs measured, side by side. This is the number that says
            # whether the calibration still describes the hardware: the
            # constants were measured on a cool device, and a phone that has
            # been generating for an hour decodes several times slower.
            # Device-side compute, reconstructed from the phone's own profile.
            # Wall time is end-to-end HTTP; on this LAN those diverge wildly
            # (1.4s of NPU decode inside a 53s round trip), and reporting only
            # the wall number would blame the accelerator for the network.
            on_device_ms = None
            if m.decode_speed_tokens_per_second and m.completion_tokens:
                on_device_ms = (m.completion_tokens / m.decode_speed_tokens_per_second) * 1000
                if m.ttft_ms:
                    on_device_ms += m.ttft_ms
                print(f"  on-device : {on_device_ms:.0f} ms compute "
                      f"({m.decode_speed_tokens_per_second:.0f} tok/s decode)"
                      f" - the rest is HTTP round trip")

            reference = on_device_ms or m.api_turnaround_latency_ms
            if selected.predicted_latency_ms and reference:
                ratio = reference / selected.predicted_latency_ms
                print(f"  predicted : {selected.predicted_latency_ms:.0f} ms"
                      + (f", {selected.predicted_energy_joules:.2f} J"
                         if selected.predicted_energy_joules else ""))
                verdict = (
                    "within expectation" if 0.5 <= ratio <= 2.0
                    else "OFF - device is not in its calibrated state"
                )
                basis = "on-device" if on_device_ms else "wall"
                print(f"  accuracy  : {ratio:.1f}x predicted latency, {basis} ({verdict})")
            print(f"  scope     : {m.energy_scope}")

    print(f"\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
