"""Fit the routing estimator heads from a completed calibration sweep.

The router needs two things it cannot compute from arithmetic alone:

    head A (quality) -- P(this device answers this prompt correctly)
    head B (length)  -- how many tokens the answer will run to

Everything else the router needs is closed-form. Latency is
``network + tokens / throughput``; energy is ``tokens x J/token``. Those
formulas do not need to be learned, they need *constants*, and this module
measures those constants from the same sweep (see ``fit_device_constants``).
That split is deliberate: recalibrating the hardware changes a number in a
table, not a model.

Both heads are k-nearest-neighbour over MiniLM embeddings of the calibration
prompts. At demo scale that beats a parametric fit on three counts: there is
no training step to get wrong, head B falls out of the same neighbour set for
free, and -- the part that matters when someone asks why -- the prediction
explains itself by naming the calibration prompts it came from.

The cost is that P(pass) is quantised to steps of 1/k, and that a prompt
unlike anything in the calibration set gets a confident-looking number
computed from irrelevant neighbours. The second problem is the dangerous one,
so ``predict`` reports a ``confidence`` derived from neighbour distance and
the router is expected to abstain from the quality gate when it is ``"low"``.

Usage:
    python benchmarks/calibration/fit_heads.py
    python benchmarks/calibration/fit_heads.py --k 7 --compare
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICES = ("phone", "pc", "cloud")

# Cosine distance to the k nearest calibration prompts, beyond which the
# quality head is not trusted. Calibrated in `choose_distance_threshold` from
# the sweep's own neighbour distances rather than guessed: anything further
# away than the worst in-domain prompt is, by construction, out of domain.
DEFAULT_CONFIDENCE_QUANTILE = 0.95


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------

def embed(texts: list[str], *, model_name: str = EMBED_MODEL, batch_size: int = 32) -> np.ndarray:
    """Mean-pooled, L2-normalised MiniLM embeddings.

    Uses transformers directly rather than sentence-transformers: the pooling
    is four lines, and it keeps the dependency surface to torch + transformers,
    both of which already have working win-arm64 wheels here.
    """

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            chunks.append(pooled.cpu().numpy())

    matrix = np.vstack(chunks).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


# --------------------------------------------------------------------------
# sweep loading
# --------------------------------------------------------------------------

def load_sweep(paths: Path | list[Path]) -> tuple[list[dict], dict[str, dict[str, dict]]]:
    """Return (ordered prompt records, {prompt_id: {device: row}}).

    Accepts several files because the sweep legs run per device and in
    parallel -- a wedged phone should not hold up the PC and cloud, and
    separate output files keep two writers off one JSONL.

    Rows that errored are dropped -- an exception tells us the tier was
    unreachable, not that it would have answered wrongly. Treating a network
    failure as a quality failure would teach head A to avoid whichever device
    happened to be flaky during the sweep.
    """

    if isinstance(paths, Path):
        paths = [paths]

    prompts: OrderedDict[str, dict] = OrderedDict()
    by_prompt: dict[str, dict[str, dict]] = {}

    lines = []
    for path in paths:
        if path.exists():
            lines.extend(path.read_text(encoding="utf-8").splitlines())

    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error") is not None:
            continue
        pid = row["id"]
        prompts.setdefault(
            pid,
            {
                "id": pid,
                "prompt": row["prompt"],
                "category": row["category"],
                "difficulty": row["difficulty"],
            },
        )
        by_prompt.setdefault(pid, {})[row["device"]] = row

    return list(prompts.values()), by_prompt


def build_label_matrices(
    prompts: list[dict], by_prompt: dict[str, dict[str, dict]], devices: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (quality, quality_mask, lengths), each shape (n_prompts, n_devices).

    ``quality_mask`` is False wherever no label exists -- a PII prompt that was
    never sent to cloud, or an item with no grader. Masked entries must be
    excluded from both fitting and scoring; counting an absent label as a
    failure is how a privacy rule silently becomes a quality signal.
    """

    n, d = len(prompts), len(devices)
    quality = np.zeros((n, d), dtype=np.float32)
    mask = np.zeros((n, d), dtype=bool)
    lengths = np.full((n, d), np.nan, dtype=np.float32)

    for i, item in enumerate(prompts):
        rows = by_prompt[item["id"]]
        for j, device in enumerate(devices):
            row = rows.get(device)
            if row is None:
                continue
            if row.get("passed") is not None:
                quality[i, j] = 1.0 if row["passed"] else 0.0
                mask[i, j] = True
            tokens = row.get("completion_tokens")
            if tokens is not None:
                lengths[i, j] = float(tokens)

    return quality, mask, lengths


# --------------------------------------------------------------------------
# device constants (the arithmetic half)
# --------------------------------------------------------------------------

def fit_device_constants(
    by_prompt: dict[str, dict[str, dict]], devices: tuple[str, ...]
) -> dict[str, dict]:
    """Measure J/token, decode throughput and TTFT per device.

    Medians throughout, not means. The first request to a cold NPU pays model
    warm-up (observed: 23.5 J and 1889 ms TTFT on the PC's first call versus
    ~0.47 J and ~100 ms after), and a thermally throttled phone call runs 3x
    slow. Those are real events but they are not the steady state the router
    should plan against, and a median ignores them without needing a rule for
    what counts as an outlier.
    """

    constants: dict[str, dict] = {}
    for device in devices:
        rows = [r[device] for r in by_prompt.values() if device in r]
        j_per_token, decode, ttft = [], [], []
        model_ids: set[str] = set()
        for row in rows:
            if row.get("model_id"):
                model_ids.add(row["model_id"])
            energy = row.get("measured_energy_joules")
            tokens = row.get("completion_tokens")
            if energy and tokens:
                j_per_token.append(energy / tokens)
            if row.get("decode_speed_tokens_per_second"):
                decode.append(row["decode_speed_tokens_per_second"])
            if row.get("ttft_ms") is not None:
                ttft.append(row["ttft_ms"])

        constants[device] = {
            # What actually answered, as reported by the server -- not what the
            # config claimed. The shipped fixtures still say "pc-model".
            "model_id": sorted(model_ids)[0] if model_ids else None,
            "energy_joules_per_token": float(np.median(j_per_token)) if j_per_token else None,
            "energy_samples": len(j_per_token),
            "decode_tokens_per_second": float(np.median(decode)) if decode else None,
            "ttft_ms": float(np.median(ttft)) if ttft else None,
            "samples": len(rows),
        }
    return constants


# --------------------------------------------------------------------------
# head A / head B: k-NN
# --------------------------------------------------------------------------

def knn_predict(
    query: np.ndarray,
    bank: np.ndarray,
    quality: np.ndarray,
    mask: np.ndarray,
    lengths: np.ndarray,
    *,
    k: int,
    exclude: int | None = None,
    similarity_floor: float = 0.0,
) -> dict:
    """Predict P(pass) and answer length for one prompt from its neighbours.

    ``exclude`` drops one bank row, which is what makes leave-one-out scoring
    honest -- without it every prompt is its own nearest neighbour and the
    reported accuracy is a memorisation check.
    """

    sims = bank @ query
    if exclude is not None:
        sims[exclude] = -np.inf

    order = np.argsort(-sims)[:k]
    neighbour_sims = sims[order]
    # Cosine similarity of L2-normalised vectors is in [-1, 1]; shift to a
    # non-negative weight so a barely-related neighbour cannot vote negatively.
    weights = np.clip(neighbour_sims, 0.0, None) + 1e-6

    p_pass, length_p50, length_p90 = {}, {}, {}
    for j in range(quality.shape[1]):
        labelled = order[mask[order, j]]
        # Mirrors CalibratedEstimator: only neighbours above the similarity
        # floor vote, so leave-one-out measures what actually ships.
        labelled = labelled[sims[labelled] >= similarity_floor]
        if len(labelled) == 0:
            p_pass[j] = None
        else:
            w = np.clip(sims[labelled], 0.0, None) + 1e-6
            p_pass[j] = float((quality[labelled, j] * w).sum() / w.sum())

        observed = lengths[order, j]
        observed = observed[~np.isnan(observed)]
        if len(observed) == 0:
            length_p50[j] = length_p90[j] = None
        else:
            length_p50[j] = float(np.percentile(observed, 50))
            length_p90[j] = float(np.percentile(observed, 90))

    mean_distance = float(1.0 - neighbour_sims.mean())
    return {
        "p_pass": p_pass,
        "length_p50": length_p50,
        "length_p90": length_p90,
        "neighbours": order.tolist(),
        "neighbour_similarity": neighbour_sims.tolist(),
        "mean_distance": mean_distance,
        "weights": weights.tolist(),
    }


def _intent_analyzer():
    """Return a callable mapping a prompt to its intent name."""

    import sys

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from peqrouter.analyzer import HeuristicPromptAnalyzer

    analyzer = HeuristicPromptAnalyzer()
    return lambda prompt: analyzer.analyze(prompt).intent.value


def choose_similarity_floor(bank: np.ndarray, *, quantile: float = 0.10) -> float:
    """Top-1 similarity below which a prompt is out of the calibration domain.

    Deliberately keyed on the *single nearest* prompt rather than the mean over
    the k nearest. Mean-over-k was tried first and is far too permissive: the
    calibration prompts are spread out enough that their own mean neighbour
    distance is large, so the resulting threshold waved through prompts whose
    best match was a similarity of 0.36 -- close enough to nothing. What makes
    a k-NN estimate meaningful is that *something* genuinely resembles the
    query, which is exactly what top-1 measures.

    The floor is the 10th percentile of the calibration set's own leave-one-out
    top-1 similarities: nine in ten calibration prompts have a better match
    than this, so anything below it is less supported than almost everything
    that was actually measured.
    """

    tops = []
    for i in range(len(bank)):
        sims = bank @ bank[i]
        sims[i] = -np.inf
        tops.append(float(sims.max()))
    return float(np.quantile(tops, quantile))


# --------------------------------------------------------------------------
# baseline: PCA + logistic regression, for the k-vs-parametric comparison
# --------------------------------------------------------------------------

def pca_fit(matrix: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    return mean, vt[:n_components].T


def logistic_fit(
    features: np.ndarray, labels: np.ndarray, *, l2: float = 1.0, steps: int = 400, lr: float = 0.5
) -> np.ndarray:
    """Plain gradient descent on the L2-regularised logistic loss."""

    x = np.hstack([features, np.ones((len(features), 1), dtype=features.dtype)])
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(steps):
        prediction = 1.0 / (1.0 + np.exp(-x @ weights))
        gradient = x.T @ (prediction - labels) / len(x)
        gradient[:-1] += l2 * weights[:-1] / len(x)  # never regularise the bias
        weights -= lr * gradient
    return weights


def logistic_predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.hstack([features, np.ones((len(features), 1), dtype=features.dtype)])
    return 1.0 / (1.0 + np.exp(-x @ weights))


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def leave_one_out(
    bank: np.ndarray,
    quality: np.ndarray,
    mask: np.ndarray,
    lengths: np.ndarray,
    devices: tuple[str, ...],
    *,
    k: int,
    similarity_floor: float = 0.0,
) -> dict:
    """Leave-one-out accuracy / Brier score for the k-NN quality head."""

    report = {}
    for j, device in enumerate(devices):
        correct = total = 0
        brier = 0.0
        base_rate = quality[mask[:, j], j].mean() if mask[:, j].any() else 0.0
        for i in range(len(bank)):
            if not mask[i, j]:
                continue
            predicted = knn_predict(
                bank[i], bank, quality, mask, lengths,
                k=k, exclude=i, similarity_floor=similarity_floor,
            )["p_pass"][j]
            if predicted is None:
                # Abstained: no neighbour close enough to vote. Scoring these
                # would measure the fallback rule, not the head.
                continue
            total += 1
            correct += int((predicted >= 0.5) == bool(quality[i, j]))
            brier += (predicted - quality[i, j]) ** 2
        report[device] = {
            "n": total,
            "accuracy": correct / total if total else None,
            "brier": brier / total if total else None,
            "base_rate": float(base_rate),
            # Always predicting the majority class. A head that cannot beat
            # this is not adding information, however good its accuracy looks.
            "majority_baseline": float(max(base_rate, 1 - base_rate)),
        }
    return report


def leave_one_out_logistic(
    bank: np.ndarray,
    quality: np.ndarray,
    mask: np.ndarray,
    devices: tuple[str, ...],
    *,
    n_components: int = 16,
) -> dict:
    report = {}
    for j, device in enumerate(devices):
        idx = np.where(mask[:, j])[0]
        if len(idx) < 10:
            report[device] = {"n": len(idx), "accuracy": None, "brier": None}
            continue
        correct = 0
        brier = 0.0
        for holdout in range(len(idx)):
            train = np.delete(idx, holdout)
            test = idx[holdout]
            mean, components = pca_fit(bank[train], n_components)
            features_train = (bank[train] - mean) @ components
            features_test = ((bank[test] - mean) @ components)[None, :]
            weights = logistic_fit(
                features_train.astype(np.float64), quality[train, j].astype(np.float64)
            )
            predicted = float(logistic_predict(features_test.astype(np.float64), weights)[0])
            correct += int((predicted >= 0.5) == bool(quality[test, j]))
            brier += (predicted - quality[test, j]) ** 2
        report[device] = {
            "n": len(idx),
            "accuracy": correct / len(idx),
            "brier": brier / len(idx),
        }
    return report


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        nargs="+",
        default=[
            Path(__file__).parent / "runs" / "sweep.jsonl",
            Path(__file__).parent / "runs" / "sweep_phone.jsonl",
            Path(__file__).parent / "runs" / "sweep_cloud_llama70b.jsonl",
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "heads")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--compare", action="store_true", help="Also score PCA+logistic baseline.")
    args = parser.parse_args()

    prompts, by_prompt = load_sweep(args.sweep)
    print(f"loaded {len(prompts)} prompts from {', '.join(str(p.name) for p in args.sweep)}")

    quality, mask, lengths = build_label_matrices(prompts, by_prompt, DEVICES)
    for j, device in enumerate(DEVICES):
        labelled = int(mask[:, j].sum())
        passes = int(quality[mask[:, j], j].sum())
        print(f"  {device:>5}: {labelled:>3} labelled, {passes:>3} passed ({passes / max(labelled, 1):.1%})")

    print(f"\nembedding with {EMBED_MODEL} ...")
    bank = embed([p["prompt"] for p in prompts])
    print(f"  embeddings: {bank.shape}")

    constants = fit_device_constants(by_prompt, DEVICES)
    print("\ndevice constants (medians):")
    for device, values in constants.items():
        energy = values["energy_joules_per_token"]
        decode = values["decode_tokens_per_second"]
        print(
            f"  {device:>5}: "
            f"{(f'{energy:.4f} J/tok' if energy else 'no energy telemetry'):<22} "
            f"{(f'{decode:.1f} tok/s' if decode else 'n/a'):<14} "
            f"n={values['samples']}"
        )

    similarity_floor = choose_similarity_floor(bank)

    print(f"\nleave-one-out, k-NN (k={args.k}):")
    knn_report = leave_one_out(
        bank, quality, mask, lengths, DEVICES, k=args.k, similarity_floor=similarity_floor
    )
    for device, values in knn_report.items():
        if values["accuracy"] is None:
            print(f"  {device:>5}: no labels")
            continue
        print(
            f"  {device:>5}: acc {values['accuracy']:.1%}  "
            f"brier {values['brier']:.3f}  "
            f"(majority baseline {values['majority_baseline']:.1%}, n={values['n']})"
        )

    comparison = None
    if args.compare:
        print("\nleave-one-out, PCA(16)+logistic:")
        comparison = leave_one_out_logistic(bank, quality, mask, DEVICES)
        for device, values in comparison.items():
            if values["accuracy"] is None:
                print(f"  {device:>5}: too few labels")
                continue
            print(f"  {device:>5}: acc {values['accuracy']:.1%}  brier {values['brier']:.3f}")

    print(f"\nout-of-domain top-1 similarity floor (q0.10): {similarity_floor:.4f}")

    # Which prompt intents were actually measured. This is the guard that does
    # the real work: embedding similarity turned out too blunt at this set size
    # to separate in-domain from out-of-domain (a haiku scored 0.37 against a
    # multiple-choice question, indistinguishable from genuine neighbours), but
    # "no creative-writing prompt was ever calibrated" is exact and checkable.
    analyzer = _intent_analyzer()
    calibrated_intents = sorted({analyzer(p["prompt"]) for p in prompts})
    print(f"calibrated intents: {', '.join(calibrated_intents)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "heads.npz",
        embeddings=bank,
        quality=quality,
        quality_mask=mask,
        lengths=lengths,
    )
    metadata = {
        "embed_model": EMBED_MODEL,
        "devices": list(DEVICES),
        "k": args.k,
        "similarity_floor": similarity_floor,
        "calibrated_intents": calibrated_intents,
        "device_constants": constants,
        "prompts": prompts,
        "evaluation": {"knn": knn_report, "logistic": comparison},
        "source_sweep": [str(p) for p in args.sweep],
    }
    (args.out_dir / "heads.json").write_text(
        # default=float: Brier scores and base rates come out of numpy as
        # float32, which json cannot serialise on its own.
        json.dumps(metadata, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    print(f"\nWrote: {args.out_dir / 'heads.npz'}\n       {args.out_dir / 'heads.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
