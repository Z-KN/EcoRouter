"""Run every calibration prompt on every tier once, and record what happened.

This is the one-time offline sweep that turns the routing heads from guesses
into measurements. For each (prompt, device) pair it records the response, the
mechanically graded pass/fail, and the observed completion length -- the two
labels the estimator heads need:

    head A (quality) <- passed
    head B (length)  <- completion_tokens

Runtime routing never does this. Every user prompt is executed exactly once,
on one device; this script exists so the router can *predict* which device
that should be.

Design notes
------------

* **Forced devices.** The router's own decision is bypassed: each prompt is
  executed on all tiers regardless of what routing would pick, because a label
  is needed for the devices routing would have *rejected*. A decision object is
  still built (executors require one) and then rewritten to target each device.
* **Resume.** Results append to JSONL and completed (id, device) pairs are
  skipped on re-run, so a phone that sleeps or drops off the LAN mid-sweep
  costs only the remaining prompts.
* **Privacy.** Items marked ``cloud_allowed: false`` are never sent to the
  cloud leg. Collecting a cloud quality label for a PII-bearing prompt would
  leak exactly what the router's privacy gate exists to prevent, and the label
  would be unusable anyway -- the gate fires before quality is ever consulted.
* **Fixed max_tokens.** All tiers get the same cap so their length labels are
  comparable. Note this right-censors the labels: a prompt that would have run
  longer is recorded at exactly the cap.

Usage:
    python benchmarks/calibration/run_sweep.py --devices phone,pc
    python benchmarks/calibration/run_sweep.py --devices cloud --max-tokens 256
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ecorouter.analyzer import HeuristicPromptAnalyzer  # noqa: E402
from ecorouter.executors import build_executors  # noqa: E402
from ecorouter.models import Device, OptimizationProfile, RouteRequest  # noqa: E402
from ecorouter.router import EcoRouter  # noqa: E402
from ecorouter.scenarios import built_in_scenarios  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graders import ANSWER_HINT, grade  # noqa: E402

_DEVICES = {"phone": Device.PHONE, "pc": Device.PC, "cloud": Device.CLOUD}

# Only answer-extraction graders need the format instruction. Appending it to a
# "write a function" prompt invites a one-line answer instead of code, which
# would measure instruction-following rather than capability.
_HINTED_GRADERS = {"number", "choice"}


def load_done(path: Path) -> set[tuple[str, str]]:
    """Return (id, device) pairs already recorded, so a re-run resumes."""

    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial final line from an interrupted run
        if row.get("error") is None:
            done.add((row["id"], row["device"]))
    return done


def build_forced_decision(router: EcoRouter, prompt: str, telemetry, device: Device, *, max_tokens: int):
    """Produce a RouteDecision targeting ``device`` regardless of routing."""

    base = router.route(RouteRequest(prompt, Device.PC, telemetry, OptimizationProfile.BALANCED))
    model_id = router.device_configs[device].model_id if hasattr(router, "device_configs") else base.model_id
    return replace(
        base,
        selected_device=device,
        model_id=model_id,
        analysis=replace(base.analysis, estimated_output_tokens=max_tokens),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="phone,pc,cloud", help="Comma-separated: phone,pc,cloud")
    parser.add_argument("--prompts", type=Path, default=Path(__file__).parent / "prompts.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs" / "sweep.jsonl")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=4, help="Retries per call (the phone returns 429 when busy).")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=15.0,
        help=(
            "Seconds to idle after a prompt fails every retry. The phone serves one "
            "request at a time and keeps generating after we give up waiting, so "
            "without this the next prompt lands on a busy device, 429s, and the run "
            "stays permanently one request behind."
        ),
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=0.0,
        help=(
            "Seconds to idle between every request. The phone needs this: it "
            "recovers from a busy state only when traffic stops entirely, so "
            "back-to-back requests let a queue build until it wedges. Retrying "
            "harder makes that worse, not better."
        ),
    )
    parser.add_argument("--limit", type=int, help="Only run the first N prompts (smoke test).")
    args = parser.parse_args()

    devices = [_DEVICES[name.strip()] for name in args.devices.split(",") if name.strip()]
    items = json.loads(args.prompts.read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)
    if done:
        print(f"resuming: {len(done)} (prompt, device) results already recorded")

    executors = build_executors(
        live_phone=Device.PHONE in devices,
        live_pc=Device.PC in devices,
        live_cloud=Device.CLOUD in devices,
    )
    router = EcoRouter(analyzer=HeuristicPromptAnalyzer())
    telemetry = built_in_scenarios()["healthy"]

    counts = {device: {"pass": 0, "fail": 0, "error": 0, "skip": 0} for device in devices}

    with args.out.open("a", encoding="utf-8") as handle:
        for device in devices:
            print(f"\n=== {device.value} ===")
            for item in items:
                key = (item["id"], device.value)
                if key in done:
                    continue
                if device is Device.CLOUD and item.get("cloud_allowed", True) is False:
                    counts[device]["skip"] += 1
                    continue

                prompt = item["prompt"]
                if item.get("grader") in _HINTED_GRADERS:
                    prompt += ANSWER_HINT

                decision = build_forced_decision(
                    router, prompt, telemetry, device, max_tokens=args.max_tokens
                )

                if args.pace > 0:
                    time.sleep(args.pace)

                observation = None
                error = None
                for attempt in range(args.retries + 1):
                    try:
                        started = time.perf_counter()
                        observation = executors[device].execute_observed(prompt, decision)
                        wall_ms = (time.perf_counter() - started) * 1000
                        error = None
                        break
                    except Exception as exc:  # noqa: BLE001 - any failure is a data point
                        error = f"{type(exc).__name__}: {exc}"
                        if attempt < args.retries:
                            # Capped exponential backoff. Starts at 4s rather than
                            # 1s because a busy phone needs to finish generating a
                            # whole response before it will accept the next one.
                            time.sleep(min(30.0, 4.0 * (2 ** attempt)))

                if observation is None:
                    counts[device]["error"] += 1
                    row = {
                        "id": item["id"],
                        "device": device.value,
                        "error": error,
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                    }
                    print(f"  {item['id']:>9} ERROR {error[:90]}")
                else:
                    passed = None
                    if item.get("quality_labeled", True) and item.get("grader"):
                        passed = grade(observation.response, item["expected"], item["grader"])
                        counts[device]["pass" if passed else "fail"] += 1
                    row = {
                        "id": item["id"],
                        "device": device.value,
                        "category": item["category"],
                        "difficulty": item["difficulty"],
                        "prompt": item["prompt"],
                        "expected": item.get("expected"),
                        "grader": item.get("grader"),
                        "passed": passed,
                        "response": observation.response,
                        "model_id": observation.model_id,
                        "prompt_tokens": observation.prompt_tokens,
                        "completion_tokens": observation.completion_tokens,
                        "max_tokens_cap": args.max_tokens,
                        "wall_latency_ms": round(wall_ms, 1),
                        "api_latency_ms": observation.api_turnaround_latency_ms,
                        "ttft_ms": observation.ttft_ms,
                        "decode_speed_tokens_per_second": observation.decode_speed_tokens_per_second,
                        "measured_energy_joules": observation.measured_energy_joules,
                        "compute_unit": observation.compute_unit,
                        "error": None,
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                    }
                    verdict = {True: "PASS", False: "FAIL", None: "----"}[passed]
                    print(
                        f"  {item['id']:>9} {verdict} "
                        f"{observation.completion_tokens or 0:>4} tok  {wall_ms:>7.0f} ms"
                    )

                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()  # crash-safe: a dropped phone loses at most one row

                if observation is None and args.cooldown > 0:
                    print(f"            cooling down {args.cooldown:.0f}s to let {device.value} drain")
                    time.sleep(args.cooldown)

    print("\n--- summary ---")
    for device, tally in counts.items():
        graded = tally["pass"] + tally["fail"]
        rate = f"{tally['pass'] / graded:.1%}" if graded else "n/a"
        print(
            f"  {device.value:>5}: pass {tally['pass']}/{graded} ({rate})  "
            f"errors {tally['error']}  privacy-skipped {tally['skip']}"
        )
    print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
