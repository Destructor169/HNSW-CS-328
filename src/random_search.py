"""Random search strategy: randomly sample parameter combinations."""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation import summarize_metrics
from src.hnsw_index import HNSWIndexWrapper


def random_search(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth_indices: np.ndarray,
    m_min: int = 4,
    m_max: int = 64,
    ef_construction_min: int = 50,
    ef_construction_max: int = 500,
    ef_search_min: int = 10,
    ef_search_max: int = 300,
    max_trials: int = 50,
    log_scale_ef: bool = True,
    k: int = 10,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run random search over parameter space.
    
    Args:
        base_vectors: Index vectors
        query_vectors: Query vectors
        ground_truth_indices: Exact k-NN indices
        m_min, m_max: M parameter range
        ef_construction_min/max: efConstruction range
        ef_search_min/max: efSearch range
        max_trials: Number of random trials
        log_scale_ef: Use log-scaling for ef parameters
        k: Number of neighbors
        seed: Random seed
        output_dir: Directory for results
    
    Returns:
        Dictionary with results and best configuration
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    results: List[Dict[str, Any]] = []
    
    def sample_param(min_val: int, max_val: int, log_scale: bool = False) -> int:
        if log_scale:
            log_min, log_max = np.log(min_val), np.log(max_val)
            return int(np.exp(np_rng.uniform(log_min, log_max)))
        return rng.randint(min_val, max_val)
    
    print(f"\n🎲 Random Search: {max_trials} trials")
    print(f"  M: [{m_min}, {m_max}], efConstruction: [{ef_construction_min}, {ef_construction_max}], "
          f"efSearch: [{ef_search_min}, {ef_search_max}]")
    
    best_score = -1.0
    best_config = None
    best_result = None
    seen_configs = set()
    
    trial = 0
    while trial < max_trials:
        # Sample parameters
        m = sample_param(m_min, m_max, log_scale=False)
        efc = sample_param(ef_construction_min, ef_construction_max, log_scale=log_scale_ef)
        efs = sample_param(ef_search_min, ef_search_max, log_scale=log_scale_ef)
        
        config_key = (m, efc, efs)
        if config_key in seen_configs:
            continue  # Skip duplicates
        seen_configs.add(config_key)
        
        trial += 1
        
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
            query_latencies_ms: List[float] = []
            all_labels = []
            for query in query_vectors:
                q_start = time.perf_counter()
                labels, _, _ = wrapper.search_hnsw(
                    queries=query.reshape(1, -1),
                    ef_search=efs,
                    k=k,
                )
                q_elapsed_ms = (time.perf_counter() - q_start) * 1000.0
                query_latencies_ms.append(q_elapsed_ms)
                all_labels.append(labels[0])
            
            predicted_labels = np.array(all_labels, dtype=np.int64)
            
            # Compute metrics
            metrics = summarize_metrics(
                predicted=predicted_labels,
                ground_truth=ground_truth_indices,
                query_time_seconds=sum(query_latencies_ms) / 1000.0,
                query_latencies_ms=query_latencies_ms,
                num_queries=len(query_vectors),
                build_time_seconds=build_time,
                memory_bytes=build_result.index_size_bytes,
                k=k,
            )
            
            # Composite score
            score = metrics.recall_at_10 + 0.1 * min(metrics.qps / 1000.0, 1.0)
            
            result = {
                "trial": trial,
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
            
            print(f"  Trial {trial:3d}/{max_trials}: "
                  f"m={m:2d} efc={efc:3d} efs={efs:3d} | "
                  f"recall@10={metrics.recall_at_10:.4f} qps={metrics.qps:.0f} score={score:.4f}")
            
        except Exception as e:
            print(f"  Trial {trial:3d}/{max_trials}: FAILED - {e}")
            continue
    
    # Convert to DataFrame and sort by score
    df = pd.DataFrame(results).sort_values("score", ascending=False)
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "random_search_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to {csv_path}")
    
    return {
        "strategy": "random_search",
        "results_df": df,
        "best_config": best_config,
        "best_score": best_score,
        "best_result": best_result,
    }
