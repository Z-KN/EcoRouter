"""Per-prompt quality and length estimates fitted from a calibration sweep.

The router's default behaviour judges quality with a *static* number: each
device carries a ``capability_score`` and every prompt is compared against the
same one. That cannot express the thing routing actually needs to know, which
is that the phone handles "what is 15% of 240" and does not handle a
multi-step word problem.

This module replaces two numbers, and only two:

    quality_sufficient   config.capability_score >= analysis.required_quality
                      -> P(this device answers *this* prompt) >= quality_floor

    estimated_output_tokens
        heuristic guess -> predicted answer length for this prompt

Everything downstream is untouched. Latency and energy are still
``network + tokens / throughput`` and ``tokens x J/token``; they just get a
measured token count instead of a guessed one. Nothing here learns a latency
or an energy -- those are arithmetic over constants measured in the sweep, and
keeping them that way is what lets the hardware be recalibrated by editing a
table instead of refitting a model.

Predictions come from the k nearest calibration prompts by cosine similarity
over MiniLM embeddings, which means every estimate can name its evidence --
see ``PromptEstimate.explain``. When the nearest calibration prompts are
further away than anything seen during calibration, ``confidence`` is
``"low"`` and the router is expected to skip the quality gate rather than act
on a number extrapolated from unrelated neighbours.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from .models import Device, ValidationError

DEFAULT_QUALITY_FLOOR = 0.5

# A device needs at least this many graded calibration prompts before its
# quality head is consulted at all. Below it, neighbour lookups mostly find no
# labelled prompt and fall through to a default -- so the head would not be
# predicting, it would be asserting. Devices under the threshold report
# ``p_pass = None`` and the router falls back to its static capability rule.
MIN_LABELS_PER_DEVICE = 20


class EstimatorUnavailableError(RuntimeError):
    """Raised when the fitted heads or their embedding backend cannot be loaded."""


@dataclass(frozen=True)
class PromptEstimate:
    """What the heads predict for one prompt, with the evidence behind it."""

    # ``None`` means "not predicted": either this device has too few
    # calibration labels overall, or no labelled neighbour was found. It does
    # not mean zero, and callers must not treat it as a failed gate.
    p_pass: Mapping[Device, float | None]
    length_p50: Mapping[Device, int]
    length_p90: Mapping[Device, int]
    confidence: str  # "high" | "low"
    mean_distance: float
    neighbours: Sequence[tuple[str, str, float]]  # (prompt_id, prompt text, similarity)

    @property
    def trusted(self) -> bool:
        """Whether the quality gate should act on these numbers at all."""

        return self.confidence == "high"

    def explain(self, device: Device) -> str:
        cited = ", ".join(f"{pid} ({sim:.2f})" for pid, _, sim in self.neighbours[:3])
        if not self.trusted:
            return (
                f"no comparable calibration prompt (nearest: {cited}); "
                f"quality gate abstained for {device.value}"
            )
        if self.p_pass.get(device) is None:
            return f"{device.value} not calibrated; quality gate fell back to static capability"
        return (
            f"P(pass) on {device.value} = {self.p_pass[device]:.2f}, "
            f"predicted {self.length_p50[device]} tokens; nearest calibration prompts: {cited}"
        )


class CalibratedEstimator:
    """k-NN quality/length heads over MiniLM embeddings of calibration prompts.

    Built by ``benchmarks/calibration/fit_heads.py``; this class only consumes
    the artifact. Loading is eager so a missing or malformed artifact fails at
    construction, where the caller can still fall back to the static default,
    rather than midway through routing a request.
    """

    def __init__(
        self,
        heads_dir: Path | str,
        *,
        quality_floor: float = DEFAULT_QUALITY_FLOOR,
        min_labels_per_device: int = MIN_LABELS_PER_DEVICE,
    ) -> None:
        if not 0.0 <= quality_floor <= 1.0:
            raise ValidationError("quality_floor must be between 0.0 and 1.0")
        self.quality_floor = float(quality_floor)
        self.min_labels_per_device = int(min_labels_per_device)

        heads_dir = Path(heads_dir)
        try:
            import numpy as np
        except ImportError as error:  # pragma: no cover - numpy is a hard dep of the heads
            raise EstimatorUnavailableError("numpy is required to load fitted heads") from error

        metadata_path = heads_dir / "heads.json"
        arrays_path = heads_dir / "heads.npz"
        if not metadata_path.exists() or not arrays_path.exists():
            raise EstimatorUnavailableError(
                f"no fitted heads in {heads_dir}; run benchmarks/calibration/fit_heads.py first"
            )

        self._meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = np.load(arrays_path)
        self._bank = arrays["embeddings"]
        self._quality = arrays["quality"]
        self._mask = arrays["quality_mask"]
        self._lengths = arrays["lengths"]
        self._np = np

        self._devices = [Device(name) for name in self._meta["devices"]]
        self._label_counts = {
            device: int(self._mask[:, index].sum())
            for index, device in enumerate(self._devices)
        }
        self.calibrated_devices = {
            device
            for device, count in self._label_counts.items()
            if count >= self.min_labels_per_device
        }
        self._k = int(self._meta["k"])
        self._similarity_floor = float(self._meta["similarity_floor"])
        self._calibrated_intents = set(self._meta.get("calibrated_intents") or ())
        self._prompts = self._meta["prompts"]
        self._embed_model = self._meta["embed_model"]

    # -- device constants measured during the same sweep ------------------

    def observed_model_id(self, device: Device) -> str | None:
        """The model that actually answered during calibration."""

        return self._meta["device_constants"][device.value].get("model_id")

    def energy_joules_per_token(self, device: Device) -> float | None:
        return self._meta["device_constants"][device.value]["energy_joules_per_token"]

    def decode_tokens_per_second(self, device: Device) -> float | None:
        return self._meta["device_constants"][device.value]["decode_tokens_per_second"]

    # -- the heads --------------------------------------------------------

    def estimate(self, prompt: str, intent: str | None = None) -> PromptEstimate:
        np = self._np
        query = self._embed(prompt)
        sims = self._bank @ query
        order = np.argsort(-sims)[: self._k]
        neighbour_sims = sims[order]
        mean_distance = float(1.0 - neighbour_sims.mean())

        p_pass: dict[Device, float] = {}
        length_p50: dict[Device, int] = {}
        length_p90: dict[Device, int] = {}

        for index, device in enumerate(self._devices):
            labelled = order[self._mask[order, index]]
            # Only neighbours that actually resemble the prompt may vote. Without
            # this, a prompt whose true nearest match is unlabelled (every PII
            # item, which is never graded) still gets a confident P(pass) from
            # whatever distant prompts happen to fill out the k slots -- observed
            # giving the phone P(pass)=1.00 on a medical summarisation from
            # neighbours at 0.26 similarity.
            labelled = labelled[sims[labelled] >= self._similarity_floor]
            if device not in self.calibrated_devices or len(labelled) == 0:
                # Not predicted. Either the device is too sparsely labelled to
                # ask, or no neighbour of this prompt carries a label for it
                # (true for cloud around the PII items, which were never sent
                # there). Reporting None makes the router fall back to its
                # static rule; reporting a number here would let a device that
                # was never measured sail through the quality gate.
                p_pass[device] = None
            else:
                weights = np.clip(sims[labelled], 0.0, None) + 1e-6
                p_pass[device] = float(
                    (self._quality[labelled, index] * weights).sum() / weights.sum()
                )

            observed = self._lengths[order, index]
            observed = observed[~np.isnan(observed)]
            if len(observed) == 0:
                length_p50[device] = length_p90[device] = 0
            else:
                length_p50[device] = int(round(float(np.percentile(observed, 50))))
                length_p90[device] = int(round(float(np.percentile(observed, 90))))

        neighbours = [
            (self._prompts[i]["id"], self._prompts[i]["prompt"], float(sims[i])) for i in order
        ]
        return PromptEstimate(
            p_pass=p_pass,
            length_p50=length_p50,
            length_p90=length_p90,
            confidence=self._confidence(float(neighbour_sims[0]), intent),
            mean_distance=mean_distance,
            neighbours=neighbours,
        )

    def _confidence(self, top_similarity: float, intent: str | None) -> str:
        """Whether the quality gate should act on this estimate.

        Two independent checks, either of which demotes to ``"low"``:

        1. **Intent was never calibrated.** Exact and decisive. The calibration
           set is built from verifiable-answer tasks, so it contains no
           creative writing; a request for a poem has no evidence behind it at
           all, whatever the embeddings say.
        2. **Nothing resembles the prompt.** Keyed on top-1 similarity against
           a floor measured from the calibration set itself.

        The first check exists because the second proved too blunt on its own
        at this set size -- a haiku scored 0.37 against a multiple-choice
        question, inside the range of genuine neighbours.
        """

        if intent is not None and self._calibrated_intents and intent not in self._calibrated_intents:
            return "low"
        return "high" if top_similarity >= self._similarity_floor else "low"

    # -- embedding --------------------------------------------------------

    def _embed(self, prompt: str):
        tokenizer, model, torch = _load_embedder(self._embed_model)
        with torch.no_grad():
            encoded = tokenizer(
                [prompt], padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vector = pooled.cpu().numpy()[0].astype("float32")
        norm = float((vector**2).sum() ** 0.5)
        return vector / max(norm, 1e-9)


# Exact snapshot already verified to load and score correctly on this
# machine's Hugging Face cache. Pinning it means the embedder never resolves
# "whatever is latest on the Hub" -- combined with local_files_only below, it
# never touches the network at all, only the weights already on disk here.
_PINNED_REVISIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}


@lru_cache(maxsize=2)
def _load_embedder(model_name: str):
    """Load the sentence encoder once per process, from local cache only.

    Cached because constructing it costs seconds and routing is expected to be
    a sub-100ms decision; paying model load per request would make the router
    slower than the inference it is trying to place. ``local_files_only``
    means this never reaches the network -- it only works because the weights
    are already downloaded to this machine's Hugging Face cache; a machine
    without that cache populated needs an online run (or ``huggingface-cli
    download``) first, which will raise ``EstimatorUnavailableError`` here
    until then.
    """

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise EstimatorUnavailableError(
            "the calibrated estimator needs torch + transformers; "
            "install them or construct PEQRouter without an estimator"
        ) from error

    revision = _PINNED_REVISIONS.get(model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            model_name, revision=revision, local_files_only=True
        ).eval()
    except OSError as error:
        raise EstimatorUnavailableError(
            f"'{model_name}' is not cached locally on this machine; the calibrated "
            "estimator only loads from the local Hugging Face cache, it does not download"
        ) from error
    return tokenizer, model, torch
