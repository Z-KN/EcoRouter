"""Score benchmark logs against ground truth.

Ground truth format: JSON array of objects with at least `id` (matching log id)
and `expected_response` (string). Example:

[
  {"id": 1, "expected_response": "It's sunny."},
  {"id": 2, "expected_response": "No personal data."}
]

Scoring currently performs simple exact-match (case-insensitive, trimmed)
accuracy. Outputs JSON summary and per-item verdicts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


GROUND_TRUTH_PATH = Path("benchmarks/ground_truth.json")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("logs", type=Path, help="Logs JSON produced by run_benchmarks.py")
    p.add_argument("--out", type=Path, help="Write scoring results to this file (optional)")
    args = p.parse_args()

    logs = json.loads(args.logs.read_text(encoding="utf-8"))
    gt_list = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    gt_by_id: Dict[int, Dict] = {int(item["id"]): item for item in gt_list}

    results: List[Dict] = []
    correct = 0
    total = 0
    skipped = 0
    for entry in logs:
        idx = int(entry.get("id"))
        if idx not in gt_by_id:
            skipped += 1
            continue
        expected = gt_by_id[idx].get("expected_response", "")
        response = entry.get("response", "")
        ok = False
        if expected and response:
            ok = normalize(expected) == normalize(response)
        results.append({"id": idx, "expected": expected, "response": response, "match": ok})
        total += 1
        if ok:
            correct += 1

    accuracy = (correct / total) if total > 0 else None
    summary = {"total": total, "correct": correct, "accuracy": accuracy, "skipped_gt_missing": skipped}

    output = {"summary": summary, "results": results}
    print(json.dumps(output, indent=2))

    # Determine output path: default is same name as logs with .scored.YYYYMMDD_HHMM.json
    if args.out:
        out_path = args.out
    else:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        stem = args.logs.stem
        out_name = f"{stem}.scored.{ts}.json"
        out_dir = Path("benchmarks/scored")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_name

    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote scoring output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
