"""Measure prompt-difficulty-estimator overhead on this CPU.

Answers one question: is a single forward pass through a small text
classifier (RouteLLM's pretrained router, or a MiniLM stand-in) negligible
next to an actual phone/PC LLM generation? It loads each candidate backend,
times repeated single-prompt inference over a short/medium/long prompt set,
and reports load time, per-prompt latency (mean/p50/p95), and process RSS
memory delta.

This script intentionally does not import the ``peqrouter`` package -- see
README.md for why estimator prototyping stays isolated from the router.

Usage:
    python bench_overhead.py --backend routellm_bert
    python bench_overhead.py --backend all --repeats 30 --out runs/my_run.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

from estimators import BACKENDS


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _percentiles(samples_ms: list[float]) -> dict:
    arr = np.array(samples_ms)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }


def bench_one_backend(name: str, prompts: list[dict], *, repeats: int, warmup: int) -> dict:
    rss_before_load = _rss_mb()
    load_start = time.perf_counter()
    estimator = BACKENDS[name]()
    load_ms = (time.perf_counter() - load_start) * 1000
    rss_after_load = _rss_mb()

    # Warm up (first calls pay for lazy kernel init / caching; excluded from
    # the reported latency so we're measuring steady-state cost).
    warmup_prompt = prompts[0]["text"]
    for _ in range(warmup):
        estimator.estimate(warmup_prompt)

    per_prompt_results = []
    all_samples_ms: list[float] = []
    for entry in prompts:
        samples_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            score = estimator.estimate(entry["text"])
            samples_ms.append((time.perf_counter() - start) * 1000)
        all_samples_ms.extend(samples_ms)
        per_prompt_results.append(
            {
                "id": entry["id"],
                "length_bucket": entry["length_bucket"],
                "char_len": len(entry["text"]),
                "last_score": round(score, 4),
                **_percentiles(samples_ms),
            }
        )

    rss_after_inference = _rss_mb()

    return {
        "backend": name,
        "load_ms": round(load_ms, 1),
        "rss_before_load_mb": round(rss_before_load, 1),
        "rss_after_load_mb": round(rss_after_load, 1),
        "rss_after_inference_mb": round(rss_after_inference, 1),
        "model_rss_delta_mb": round(rss_after_load - rss_before_load, 1),
        "per_prompt": per_prompt_results,
        "overall": _percentiles(all_samples_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=(*BACKENDS.keys(), "all"),
        default="all",
        help="Which estimator backend to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=20, help="Timed calls per prompt.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warm-up calls.")
    parser.add_argument("--prompts", type=Path, default=Path(__file__).parent / "prompts.json")
    parser.add_argument("--out", type=Path, help="Write JSON results here.")
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    backend_names = list(BACKENDS.keys()) if args.backend == "all" else [args.backend]

    results = []
    for name in backend_names:
        print(f"--- {name} ---")
        result = bench_one_backend(name, prompts, repeats=args.repeats, warmup=args.warmup)
        results.append(result)
        print(f"load: {result['load_ms']} ms, model RSS: +{result['model_rss_delta_mb']} MB")
        for row in result["per_prompt"]:
            print(
                f"  {row['id']:>10} ({row['length_bucket']:>6}, {row['char_len']:>3} chars): "
                f"mean={row['mean_ms']:.2f}ms p50={row['p50_ms']:.2f}ms p95={row['p95_ms']:.2f}ms"
            )
        print(f"  overall: mean={result['overall']['mean_ms']:.2f}ms p95={result['overall']['p95_ms']:.2f}ms")
        print()

    if args.out:
        out_path = args.out
    else:
        out_dir = Path(__file__).parent / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = out_dir / f"overhead.{ts}.json"

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
