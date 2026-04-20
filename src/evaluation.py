from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class EvaluationResult:
    """Comprehensive evaluation metrics for HNSW experiments."""
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    mrr_at_k: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    qps: float
    build_time_seconds: float
    memory_mb: float
    
    # Backward compatibility
    @property
    def recall(self) -> float:
        return self.recall_at_10
    
    @property
    def latency_per_query_ms(self) -> float:
        return self.latency_p50_ms
    
    @property
    def memory_bytes(self) -> int:
        """Convert memory_mb back to bytes."""
        return int(self.memory_mb * 1024 * 1024)

    def to_dict(self) -> Dict[str, float]:
        return {
            "recall_at_1": float(self.recall_at_1),
            "recall_at_10": float(self.recall_at_10),
            "recall_at_100": float(self.recall_at_100),
            "mrr_at_k": float(self.mrr_at_k),
            "latency_p50_ms": float(self.latency_p50_ms),
            "latency_p95_ms": float(self.latency_p95_ms),
            "latency_p99_ms": float(self.latency_p99_ms),
            "qps": float(self.qps),
            "build_time_s": float(self.build_time_seconds),
            "memory_mb": float(self.memory_mb),
        }


def compute_recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute average recall@k over all queries."""
    if predicted.shape[0] != ground_truth.shape[0]:
        raise ValueError("predicted and ground_truth must contain same number of queries")

    use_k = min(k, predicted.shape[1], ground_truth.shape[1])
    if use_k <= 0:
        raise ValueError("k must be > 0")

    total = 0.0
    for pred, gt in zip(predicted[:, :use_k], ground_truth[:, :use_k], strict=False):
        total += len(set(pred.tolist()) & set(gt.tolist())) / use_k
    return total / predicted.shape[0]


def compute_mrr_at_k(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute Mean Reciprocal Rank: average 1/rank of ground_truth[i,0] in predicted."""
    if predicted.shape[0] != ground_truth.shape[0]:
        raise ValueError("predicted and ground_truth must contain same number of queries")
    
    use_k = min(k, predicted.shape[1])
    rr_sum = 0.0
    for pred, gt in zip(predicted[:, :use_k], ground_truth):
        target = gt[0]  # True top-1 neighbor
        try:
            rank = int(np.where(pred == target)[0][0]) + 1
            rr_sum += 1.0 / rank
        except IndexError:
            rr_sum += 0.0  # Not found in top-k
    return rr_sum / predicted.shape[0]


def compute_percentile(values: List[float], p: float) -> float:
    """Compute p-th percentile of latency values."""
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def summarize_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    query_time_seconds: float,
    query_latencies_ms: List[float],
    num_queries: int,
    build_time_seconds: float,
    memory_bytes: int,
    k: int = 10,
) -> EvaluationResult:
    """Compute all evaluation metrics."""
    recall_at_1 = compute_recall_at_k(predicted, ground_truth, k=1)
    recall_at_10 = compute_recall_at_k(predicted, ground_truth, k=min(10, k))
    recall_at_100 = compute_recall_at_k(predicted, ground_truth, k=min(100, k))
    mrr = compute_mrr_at_k(predicted, ground_truth, k=k)
    
    # Latency metrics
    latency_p50 = compute_percentile(query_latencies_ms, 50)
    latency_p95 = compute_percentile(query_latencies_ms, 95)
    latency_p99 = compute_percentile(query_latencies_ms, 99)
    
    # Throughput: queries per second
    query_total_seconds = sum(query_latencies_ms) / 1000.0 if query_latencies_ms else query_time_seconds
    qps = num_queries / max(query_total_seconds, 0.001)
    
    # Memory in MB
    memory_mb = memory_bytes / (1024.0 * 1024.0)
    
    return EvaluationResult(
        recall_at_1=recall_at_1,
        recall_at_10=recall_at_10,
        recall_at_100=recall_at_100,
        mrr_at_k=mrr,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        latency_p99_ms=latency_p99,
        qps=qps,
        build_time_seconds=build_time_seconds,
        memory_mb=memory_mb,
    )
