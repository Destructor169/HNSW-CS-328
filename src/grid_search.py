"""Grid search strategy: evaluate all combinations of parameters."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import summarize_metrics
from src.hnsw_index import HNSWIndexWrapper


def grid_search(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth_indices: np.ndarray,
    m_values: list[int],
    ef_construction_values: list[int],
    ef_search_values: list[int],
    k: int = 10,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run grid search over parameter combinations.
    
    Args:
        base_vectors: Index vectors (n_base, dim)
        query_vectors: Query vectors (n_queries, dim)
        ground_truth_indices: Exact k-NN indices (n_queries, k)
        m_values: List of M values to test
        ef_construction_values: List of efConstruction values
        ef_search_values: List of efSearch values
        k: Number of neighbors
        output_dir: Directory for results CSV
    
    Returns:
        Dictionary with results and best configuration
    """
    results: list[dict[str, Any]] = []
    
    # Generate all combinations
    combinations = [
        (m, efc, efs)
        for m in m_values
        for efc in ef_construction_values
        for efs in ef_search_values
    ]
    
    print(f"\n🔍 Grid Search: {len(combinations)} combinations")
    print(f"  M: {m_values}, efConstruction: {ef_construction_values}, efSearch: {ef_search_values}")
    
    best_score = -1.0
    best_config = None
    best_result = None
    
    for trial_idx, (m, efc, efs) in enumerate(combinations, 1):
        try:
            # Build index
            wrapper = HNSWIndexWrapper(space="l2", dimension=base_vectors.shape[1])
            build_start = time.perf_counter()
            build_result = wrapper.build_hnsw_index(
                data=base_vectors,
                m=m,
                ef_construction=efc,
            )
            build_time = time.perf_counter() - build_start
            
            # Query with per-query latency measurement
            query_latencies_ms: list[float] = []
            labels_list = []
            for query in query_vectors:
                start_time = time.perf_counter()
                labels, _, _ = wrapper.search_hnsw(
                    queries=query.reshape(1, -1),
                    ef_search=efs,
                    k=k,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000
                query_latencies_ms.append(latency_ms)
                labels_list.append(labels[0])
            
            labels = np.array(labels_list)
            query_time = sum(query_latencies_ms) / 1000.0
            
            # Compute metrics
            metrics = summarize_metrics(
                predicted=labels,
                ground_truth=ground_truth_indices,
                query_time_seconds=query_time,
                query_latencies_ms=query_latencies_ms,
                num_queries=len(query_vectors),
                build_time_seconds=build_time,
                memory_bytes=build_result.index_size_bytes,
                k=k,
            )
            
            # Composite score: recall@10 + 0.1 * min(QPS/1000, 1.0)
            score = metrics.recall_at_10 + 0.1 * min(metrics.qps / 1000.0, 1.0)
            
            result = {
                "trial": trial_idx,
                "m": m,
                "ef_construction": efc,
                "ef_search": efs,
                "recall_at_1": metrics.recall_at_1,
                "recall_at_10": metrics.recall_at_10,
                "recall_at_100": metrics.recall_at_100,
                "mrr_at_k": metrics.mrr_at_k,
                "latency_p50_ms": metrics.latency_p50_ms,
                "latency_p95_ms": metrics.latency_p95_ms,
                "latency_p99_ms": metrics.latency_p99_ms,
                "qps": metrics.qps,
                "build_time_s": metrics.build_time_seconds,
                "memory_mb": metrics.memory_mb,
                "score": score,
            }
            results.append(result)
            
            if score > best_score:
                best_score = score
                best_config = {"m": m, "ef_construction": efc, "ef_search": efs}
                best_result = result
            
            print(f"  Trial {trial_idx:3d}/{len(combinations)}: "
                  f"m={m:2d} efc={efc:3d} efs={efs:3d} | "
                  f"recall@10={metrics.recall_at_10:.4f} qps={metrics.qps:.0f} score={score:.4f}")
            
        except Exception as e:
            print(f"  Trial {trial_idx:3d}/{len(combinations)}: FAILED - {e}")
            continue
    
    # Convert to DataFrame and sort by score
    df = pd.DataFrame(results).sort_values("score", ascending=False)
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "grid_search_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to {csv_path}")
    
    return {
        "strategy": "grid_search",
        "results_df": df,
        "best_config": best_config,
        "best_score": best_score,
        "best_result": best_result,
    }
