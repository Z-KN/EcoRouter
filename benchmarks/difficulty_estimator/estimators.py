"""Candidate prompt-difficulty estimators for the overhead benchmark.

Shaped like ``peqrouter.analyzer.PromptAnalyzer`` on purpose: if one of these
backends earns its way into the real router later, the interface should
already fit -- but nothing here imports ``peqrouter``, and nothing in
``peqrouter`` imports this. See ../README.md for why the two packages stay
isolated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class DifficultyEstimator(Protocol):
    name: str

    def estimate(self, prompt: str) -> float:
        """Return a difficulty/routing scalar for one prompt."""


@dataclass
class RouteLLMBertEstimator:
    """RouteLLM's pretrained router: fine-tuned xlm-roberta-base, 3-way
    classifier (weak model win / tie / strong model win). We report
    P(strong model needed) = P(label 2) as the difficulty scalar.
    """

    checkpoint: str = "routellm/bert_gpt4_augmented"
    name: str = "routellm_bert"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint)
        self.model.eval()
        torch.set_num_threads(1)  # isolate single-core latency; see README

    @torch.inference_mode()
    def estimate(self, prompt: str) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, -1])


@dataclass
class MiniLMPlaceholderEstimator:
    """MiniLM-L6 encoder + an untrained linear head standing in for a future
    logistic-regression/GBT classifier fit on calibration data. The head is
    a few hundred FLOPs -- irrelevant to timing -- so this measures what
    matters: the encoder forward pass.
    """

    checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = "minilm_l6_placeholder"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModel.from_pretrained(self.checkpoint)
        self.model.eval()
        torch.set_num_threads(1)
        self.head = torch.nn.Linear(self.model.config.hidden_size, 1)

    @torch.inference_mode()
    def estimate(self, prompt: str) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        output = self.model(**inputs)
        pooled = output.last_hidden_state[:, 0]  # CLS-equivalent pooling
        return float(torch.sigmoid(self.head(pooled))[0, 0])


BACKENDS: dict[str, type[DifficultyEstimator]] = {
    "routellm_bert": RouteLLMBertEstimator,
    "minilm_l6_placeholder": MiniLMPlaceholderEstimator,
}
