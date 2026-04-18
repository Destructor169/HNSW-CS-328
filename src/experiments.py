from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.evaluation import summarize_metrics
from src.hnsw_index import HNSWIndexWrapper


@dataclass
class ExperimentConfig:
    k: int = 10
    default_m: int = 16
    default_ef_construction: int = 200
    default_ef_search: int = 100


def run_single_experiment(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth_indices: np.ndarray,
    m: int,
    ef_construction: int,
    ef_search: int,
    k: int,
) -> Dict[str, float | int]:
    index = HNSWIndexWrapper(dimension=base_vectors.shape[1])
    build_result = index.build_hnsw_index(
        data=base_vectors,
        m=m,
        ef_construction=ef_construction,
    )
    labels, _, query_time = index.search_hnsw(queries=query_vectors, ef_search=ef_search, k=k)

    metrics = summarize_metrics(
        predicted=labels,
        ground_truth=ground_truth_indices,
        query_time_seconds=query_time,
        num_queries=query_vectors.shape[0],
        build_time_seconds=build_result.build_time_seconds,
        memory_bytes=build_result.index_size_bytes,
        k=k,
    )

    return {
        "m": m,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "recall": metrics.recall,
        "latency_ms": metrics.latency_per_query_ms,
        "build_time_s": metrics.build_time_seconds,
        "memory_bytes": metrics.memory_bytes,
    }


def run_parameter_sweeps(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth_indices: np.ndarray,
    config: ExperimentConfig,
) -> pd.DataFrame:
    records: List[Dict[str, float | int | str]] = []

    for m in [8, 16, 24, 32, 48, 64]:
        result = run_single_experiment(
            base_vectors,
            query_vectors,
            ground_truth_indices,
            m=m,
            ef_construction=config.default_ef_construction,
            ef_search=config.default_ef_search,
            k=config.k,
        )
        result["experiment"] = "effect_of_m"
        records.append(result)

    for ef_construction in [50, 100, 200, 300, 400, 500]:
        result = run_single_experiment(
            base_vectors,
            query_vectors,
            ground_truth_indices,
            m=config.default_m,
            ef_construction=ef_construction,
            ef_search=config.default_ef_search,
            k=config.k,
        )
        result["experiment"] = "effect_of_ef_construction"
        records.append(result)

    for ef_search in [10, 30, 50, 100, 150, 200, 300]:
        result = run_single_experiment(
            base_vectors,
            query_vectors,
            ground_truth_indices,
            m=config.default_m,
            ef_construction=config.default_ef_construction,
            ef_search=ef_search,
            k=config.k,
        )
        result["experiment"] = "effect_of_ef_search"
        records.append(result)

    return pd.DataFrame.from_records(records)


def compare_default_vs_learned(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth_indices: np.ndarray,
    learned_params: Dict[str, int],
    config: ExperimentConfig,
) -> pd.DataFrame:
    default_result = run_single_experiment(
        base_vectors,
        query_vectors,
        ground_truth_indices,
        m=config.default_m,
        ef_construction=config.default_ef_construction,
        ef_search=config.default_ef_search,
        k=config.k,
    )
    default_result["configuration"] = "default"

    learned_result = run_single_experiment(
        base_vectors,
        query_vectors,
        ground_truth_indices,
        m=int(learned_params["m"]),
        ef_construction=int(learned_params["ef_construction"]),
        ef_search=int(learned_params["ef_search"]),
        k=config.k,
    )
    learned_result["configuration"] = "learned"

    return pd.DataFrame([default_result, learned_result])


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
