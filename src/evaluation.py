from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class EvaluationResult:
    recall: float
    latency_per_query_ms: float
    build_time_seconds: float
    memory_bytes: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "recall": float(self.recall),
            "latency_ms": float(self.latency_per_query_ms),
            "build_time_s": float(self.build_time_seconds),
            "memory_bytes": float(self.memory_bytes),
        }


def compute_recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    if predicted.shape[0] != ground_truth.shape[0]:
        raise ValueError("predicted and ground_truth must contain same number of queries")

    use_k = min(k, predicted.shape[1], ground_truth.shape[1])
    if use_k <= 0:
        raise ValueError("k must be > 0")

    total = 0.0
    for pred, gt in zip(predicted[:, :use_k], ground_truth[:, :use_k], strict=False):
        total += len(set(pred.tolist()) & set(gt.tolist())) / use_k
    return total / predicted.shape[0]


def summarize_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    query_time_seconds: float,
    num_queries: int,
    build_time_seconds: float,
    memory_bytes: int,
    k: int,
) -> EvaluationResult:
    recall = compute_recall_at_k(predicted, ground_truth, k=k)
    latency_ms = (query_time_seconds / max(1, num_queries)) * 1000.0
    return EvaluationResult(
        recall=recall,
        latency_per_query_ms=latency_ms,
        build_time_seconds=build_time_seconds,
        memory_bytes=memory_bytes,
    )
