from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class ExactKNNResult:
    distances: np.ndarray
    indices: np.ndarray


def compute_exact_neighbors(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
) -> ExactKNNResult:
    """Run brute-force nearest-neighbor search for ground truth."""
    if k <= 0:
        raise ValueError("k must be > 0")

    model = NearestNeighbors(
        n_neighbors=k,
        algorithm="brute",
        metric=metric,
        n_jobs=-1,
    )
    model.fit(base_vectors)
    distances, indices = model.kneighbors(query_vectors)
    return ExactKNNResult(
        distances=distances.astype(np.float32, copy=False),
        indices=indices.astype(np.int64, copy=False),
    )


def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute average Recall@k over all queries."""
    if predicted.shape[0] != ground_truth.shape[0]:
        raise ValueError("predicted and ground_truth must have the same number of queries")

    use_k = min(k, predicted.shape[1], ground_truth.shape[1])
    if use_k <= 0:
        raise ValueError("k must be > 0 and less than available neighbors")

    hits = 0
    for pred_row, gt_row in zip(predicted[:, :use_k], ground_truth[:, :use_k], strict=False):
        hits += len(set(pred_row.tolist()) & set(gt_row.tolist()))

    return hits / (predicted.shape[0] * use_k)
