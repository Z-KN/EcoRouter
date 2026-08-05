Benchmarks
==========

This folder contains small, self-contained benchmarking helpers for offline
experiments. The scripts intentionally avoid importing the live `ecorouter`
package so they can run without Presidio or Cirrascale SDKs.

Files
-----

- `run_benchmarks.py`: lightweight harness that computes profile-weighted scores
  for each device given a telemetry snapshot and a model catalog. Outputs JSON
  or CSV results suitable for comparison against ground truth.
- `ground_truth_template.json`: a template to record expected winners for
  scenarios/profiles.

Usage
-----

Run the harness against an examples telemetry snapshot and models catalog:

```bash
python3 benchmarks/run_benchmarks.py \
  --telemetry examples/telemetry/healthy.json \
  --models examples/models.json \
  --input-tokens 8 --output-tokens 20 \
  --out results.json
```

The script prints a concise table and writes `results.json` when `--out` is
provided.

Integrate with CI or perf experiments by adding a `ground_truth.json` file and
comparing output programmatically.
