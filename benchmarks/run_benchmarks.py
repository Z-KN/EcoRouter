"""Run offline benchmarks and record model outputs.

This harness supports two execution modes:
- `baseline`: send prompts to a baseline model endpoint (HTTP) or run a local
    simulated baseline function.
- `routed`: send prompts to a routed stack endpoint (HTTP) or run a local
    simulated router that selects a device and model then returns a response.

The script logs a JSON array of records containing prompt id, response, timing,
and routing metadata. Use `benchmarks/score_logs.py` to evaluate logs against a
ground-truth JSON file.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import time


@dataclass
class DeviceConfig:
    model_id: str
    capability_score: float


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


_PROFILE_WEIGHTS = {
    "balanced": {
        "latency": 0.30,
        "energy": 0.20,
        "utilization": 0.10,
        "thermal": 0.10,
        "battery": 0.10,
        "cost": 0.10,
        "quality": 0.10,
    },
    "low-latency": {
        "latency": 0.55,
        "energy": 0.10,
        "utilization": 0.10,
        "thermal": 0.05,
        "battery": 0.05,
        "cost": 0.05,
        "quality": 0.10,
    },
    "energy-saver": {
        "latency": 0.15,
        "energy": 0.40,
        "utilization": 0.10,
        "thermal": 0.15,
        "battery": 0.10,
        "cost": 0.05,
        "quality": 0.05,
    },
    "high-quality": {
        "latency": 0.15,
        "energy": 0.10,
        "utilization": 0.05,
        "thermal": 0.05,
        "battery": 0.05,
        "cost": 0.10,
        "quality": 0.50,
    },
}


def compute_score(
    profile: str,
    telemetry: Dict,
    config: DeviceConfig,
    total_tokens: int,
    network_latency_ms: float,
) -> Tuple[Optional[float], Dict[str, float], Optional[float], Optional[float], Optional[float]]:
    can_estimate = telemetry.get("throughput_tokens_per_second", 0) > 0
    if not can_estimate:
        return None, {}, None, None, None

    latency = network_latency_ms + total_tokens / telemetry["throughput_tokens_per_second"] * 1000
    energy = total_tokens * telemetry["energy_joules_per_token"]
    cloud_cost = total_tokens / 1000 * telemetry.get("cloud_cost_per_1k_tokens_usd", 0)

    penalties = {
        "latency": _clamp(latency / 10_000),
        "energy": _clamp(energy / 50),
        "utilization": telemetry.get("utilization", 0.0),
        "thermal": telemetry.get("thermal_pressure", 0.0),
        "cost": _clamp(cloud_cost / 0.10),
        "quality": 1.0 - config.capability_score,
    }
    if telemetry.get("battery_percent") is not None:
        penalties["battery"] = 1.0 - telemetry["battery_percent"] / 100

    weights = _PROFILE_WEIGHTS[profile]
    applicable_weight = sum(weights[name] for name in penalties)
    score = sum(weights[name] * value for name, value in penalties.items()) / applicable_weight
    return score, penalties, latency, energy, cloud_cost


def load_models(path: Path) -> Dict[str, DeviceConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for name, info in payload.get("devices", {}).items():
        result[name] = DeviceConfig(model_id=info["model_id"], capability_score=info["capability_score"])
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--input-tokens", type=int, default=8)
    p.add_argument("--output-tokens", type=int, default=20)
    p.add_argument("--profile", choices=tuple(_PROFILE_WEIGHTS.keys()), help="If set, run only this profile")
    p.add_argument("--out", type=Path, help="Write JSON results to this file")
    p.add_argument("--prompts", type=Path, help="File with prompts (one per line) or JSON array")
    p.add_argument("--mode", choices=("baseline", "routed", "both"), default="both")
    args = p.parse_args()

    telemetry = json.loads(args.telemetry.read_text(encoding="utf-8"))
    models = load_models(args.models)
    total_tokens = args.input_tokens + args.output_tokens
    # load prompts
    prompts: list[str] = []
    if args.prompts:
        raw = args.prompts.read_text(encoding="utf-8")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                prompts = [str(x) for x in parsed]
            else:
                raise ValueError("prompts JSON must be an array")
        except json.JSONDecodeError:
            prompts = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        # default small prompt set
        prompts = [
            "What's the weather tomorrow?",
            "Summarize the profile for John Smith",
            "Translate to French: Hello, how are you?",
        ]

    results = []
    # helper executors
    def simulated_baseline(prompt: str) -> Dict:
        # simple deterministic simulated response
        return {"model_id": "baseline-model", "response": f"baseline: {prompt[:200]}"}

    def simulated_routed(prompt: str) -> Dict:
        # very simple routing heuristic: long prompts -> cloud, medium -> pc, short -> phone
        l = len(prompt)
        if l > 200:
            device = "cloud"
            model = models["cloud"].model_id if "cloud" in models else "cloud-model"
        elif l > 80:
            device = "pc"
            model = models.get("pc", DeviceConfig("pc-model", 0.8)).model_id
        else:
            device = "phone"
            model = models.get("phone", DeviceConfig("phone-model", 0.6)).model_id
        return {"selected_device": device, "model_id": model, "response": f"{device} simulated: {prompt[:200]}"}

    # endpoints will be configured later; current harness uses simulated executors only

    modes = (args.mode,) if args.mode != "both" else ("baseline", "routed")
    for mode in modes:
        for idx, prompt in enumerate(prompts, start=1):
            entry = {
                "id": idx,
                "mode": mode,
                "prompt": prompt,
            }

            start = time.time()
            if mode == "baseline":
                data = simulated_baseline(prompt)
                entry.update({"model_id": data["model_id"], "response": data["response"], "status": "ok"})
            else:
                data = simulated_routed(prompt)
                entry.update({"selected_device": data["selected_device"], "model_id": data["model_id"], "response": data["response"], "status": "ok"})

            runtime_ms = (time.time() - start) * 1000.0
            entry["runtime_ms"] = runtime_ms
            # token bookkeeping
            entry["input_tokens"] = args.input_tokens
            entry["output_tokens"] = args.output_tokens
            entry["total_tokens"] = args.input_tokens + args.output_tokens
            results.append(entry)

    print(json.dumps(results, indent=2))

    # Default output folder for runs
    if args.out:
        out_path = args.out
    else:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_dir = Path("benchmarks/runs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"run.{ts}.json"

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote run output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
