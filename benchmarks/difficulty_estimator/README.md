Difficulty Estimator Overhead Benchmark
========================================

Prototyping area for the routing difficulty estimator discussed for the
EcoRouter policy: given a prompt, predict how much "quality headroom" it
needs before touching any candidate model, so the router can pick phone/PC/
cloud without guessing blind or running every tier.

This measures one narrow question first: **is the estimator's own overhead
negligible** next to an actual phone/PC LLM generation? It is not yet wired
into `ecorouter` -- see "Status" below.

Why this lives outside `ecorouter/`
------------------------------------

The core `ecorouter` package is meant to run wherever the router runs today
(the PC/X-Elite side, per `RouteRequest.origin` and the CLI) and, per
`pyproject.toml`, has no external dependencies at all -- the privacy analyzer
is pure regex. Bringing in `torch`/`transformers` just to *evaluate* candidate
estimators would force that weight onto every install. This folder stays
self-contained -- like `benchmarks/run_benchmarks.py` -- so it can be deleted,
swapped, or extended for other devices without touching the router package.

Runtime note: win-arm64 (this machine -- confirmed Snapdragon/ARM64, likely
the actual X-Elite target)
------------------------------------------------------------------------------

`torch` has no stable PyPI wheel for win-arm64 yet, so `requirements.txt`
pulls the CPU **nightly** build from PyTorch's own index, which does publish
one. That's fine for this one-off verification script, but it's *not* the
recommended runtime for an eventual on-device estimator:

- `onnxruntime` already ships a stable win-arm64 wheel today, plus an Android
  AAR for the phone and a QNN execution provider if NPU is ever revisited --
  one runtime story across devices instead of one per platform.
- RouteLLM's pretrained checkpoint (`routellm/bert_gpt4_augmented`) is only
  published as PyTorch safetensors (no ONNX export on the Hub as of writing),
  which is why this benchmark uses torch directly rather than exporting to
  ONNX first.

If this estimator graduates past prototyping, plan to export the winning
checkpoint to ONNX (on a machine with a stable torch install) and re-run
these same prompts through an ONNX Runtime backend for a fair before/after
comparison, rather than assuming torch-nightly numbers hold for the eventual
deployment runtime.

Files
-----

- `estimators.py` -- candidate backends behind a shared `estimate(prompt) ->
  float` interface (shaped like `ecorouter.analyzer.PromptAnalyzer` on
  purpose, in case one graduates later):
  - `routellm_bert`: RouteLLM's actual pretrained router
    (`routellm/bert_gpt4_augmented`, fine-tuned `xlm-roberta-base`). Note its
    vocab is 250k (multilingual) vs English BERT-base's ~30k, so it's
    roughly 278M params on disk, not the ~110M a plain "BERT-base" estimate
    would suggest -- vocab size inflates the embedding table, not per-token
    compute, so latency is still governed by the 12-layer/768-hidden
    transformer body.
  - `minilm_l6_placeholder`: `sentence-transformers/all-MiniLM-L6-v2` +
    an untrained linear head standing in for a future logistic-regression/
    GBT classifier fit on calibration data -- the head is a few hundred
    FLOPs, irrelevant to timing. This is the fallback if `routellm_bert`'s
    score doesn't transfer well to our phone/PC/cloud gap (see prior
    discussion) or if its size is a problem for the phone later.
- `bench_overhead.py` -- loads a backend, warms it up, times `--repeats`
  calls per prompt across a short/medium/long prompt set, and reports load
  time, per-prompt latency (mean/p50/p95), and process RSS memory delta.
  Single-threaded (`torch.set_num_threads(1)`) to measure worst-case
  single-core latency rather than letting it soak up all cores.
- `prompts.json` -- 8 prompts across 3 length buckets (short/medium/long),
  chosen so latency-vs-prompt-length can be inspected, not just an average.
- `runs/` -- JSON output, one file per invocation.

Usage
-----

```bash
pip install -r benchmarks/difficulty_estimator/requirements.txt
python benchmarks/difficulty_estimator/bench_overhead.py --backend all
```

Status / what this does *not* do yet
-------------------------------------

This only checks overhead. It does not check whether `routellm_bert`'s score
actually correlates with phone/PC/cloud pass-fail on our prompts (that needs
the calibration dataset + `benchmarks/run_benchmarks.py`-style scoring
against ground truth -- see `benchmarks/ground_truth_template.json` for the
existing convention). Nothing here is imported by `ecorouter`, and nothing
in `ecorouter` imports this.
