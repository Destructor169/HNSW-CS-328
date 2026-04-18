from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd

from src.evaluation import summarize_metrics
from src.hnsw_index import HNSWIndexWrapper


@dataclass
class OptimizationConfig:
    trials: int = 50
    k: int = 10
    m_min: int = 4
    m_max: int = 64
    ef_construction_min: int = 50
    ef_construction_max: int = 500
    ef_search_min: int = 10
    ef_search_max: int = 300
    latency_weight: float = 0.001


@dataclass
class MultiObjectiveConfig:
    enabled: bool = True
    trials: int = 40
    k: int = 10
    m_min: int = 4
    m_max: int = 64
    ef_construction_min: int = 50
    ef_construction_max: int = 500
    ef_search_min: int = 10
    ef_search_max: int = 300
    score_recall_weight: float = 0.5
    score_latency_weight: float = 0.25
    score_build_time_weight: float = 0.15
    score_memory_weight: float = 0.10
    min_recall_for_ranking: float = 0.99


def _run_single_config(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth_indices: np.ndarray,
    m: int,
    ef_construction: int,
    ef_search: int,
    k: int,
) -> Dict[str, float | int]:
    wrapper = HNSWIndexWrapper(dimension=base_vectors.shape[1])
    build_result = wrapper.build_hnsw_index(
        data=base_vectors,
        m=m,
        ef_construction=ef_construction,
    )
    labels, _, query_time = wrapper.search_hnsw(
        queries=query_vectors,
        ef_search=ef_search,
        k=k,
    )

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


def optimize_hnsw_parameters(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth_indices: np.ndarray,
    config: OptimizationConfig,
    seed: int = 42,
) -> Tuple[optuna.Study, List[Dict[str, Any]]]:
    """Optimize HNSW parameters using Optuna's TPE sampler."""
    history: List[Dict[str, Any]] = []

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        m = trial.suggest_int("m", config.m_min, config.m_max)
        ef_construction = trial.suggest_int(
            "ef_construction",
            config.ef_construction_min,
            config.ef_construction_max,
        )
        ef_search = trial.suggest_int("ef_search", config.ef_search_min, config.ef_search_max)

        result = _run_single_config(
            base_vectors=base_vectors,
            query_vectors=query_vectors,
            ground_truth_indices=ground_truth_indices,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            k=config.k,
        )

        score = float(result["recall"]) - config.latency_weight * float(result["latency_ms"])

        trial.set_user_attr("recall", float(result["recall"]))
        trial.set_user_attr("latency_ms", float(result["latency_ms"]))
        trial.set_user_attr("build_time_s", float(result["build_time_s"]))
        trial.set_user_attr("memory_bytes", int(result["memory_bytes"]))

        result["score"] = score
        history.append(result)
        return score

    study.optimize(objective, n_trials=config.trials)
    return study, history


def optimize_hnsw_parameters_multi_objective(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth_indices: np.ndarray,
    config: MultiObjectiveConfig,
    seed: int = 42,
) -> Tuple[optuna.Study, List[Dict[str, Any]]]:
    """Optimize for recall, latency, build time and memory simultaneously."""
    history: List[Dict[str, Any]] = []

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize", "minimize"],
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        m = trial.suggest_int("m", config.m_min, config.m_max)
        ef_construction = trial.suggest_int(
            "ef_construction",
            config.ef_construction_min,
            config.ef_construction_max,
        )
        ef_search = trial.suggest_int("ef_search", config.ef_search_min, config.ef_search_max)

        result = _run_single_config(
            base_vectors=base_vectors,
            query_vectors=query_vectors,
            ground_truth_indices=ground_truth_indices,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            k=config.k,
        )

        recall = float(result["recall"])
        latency_ms = float(result["latency_ms"])
        build_time_s = float(result["build_time_s"])
        memory_bytes = float(result["memory_bytes"])

        trial.set_user_attr("recall", recall)
        trial.set_user_attr("latency_ms", latency_ms)
        trial.set_user_attr("build_time_s", build_time_s)
        trial.set_user_attr("memory_bytes", int(memory_bytes))

        history.append(
            {
                "trial_number": trial.number,
                "m": m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
                "recall": recall,
                "latency_ms": latency_ms,
                "build_time_s": build_time_s,
                "memory_bytes": int(memory_bytes),
            }
        )

        return recall, latency_ms, build_time_s, memory_bytes

    study.optimize(objective, n_trials=config.trials)
    return study, history


def build_ranked_pareto_report(
    study: optuna.Study,
    recall_weight: float = 0.5,
    latency_weight: float = 0.25,
    build_time_weight: float = 0.15,
    memory_weight: float = 0.10,
    min_recall_for_ranking: float = 0.99,
) -> pd.DataFrame:
    """Build a ranked Pareto-front report from a multi-objective Optuna study."""
    if not study.best_trials:
        return pd.DataFrame(
            columns=[
                "pareto_rank",
                "trial_number",
                "m",
                "ef_construction",
                "ef_search",
                "recall",
                "latency_ms",
                "build_time_s",
                "memory_bytes",
                "aggregate_score",
            ]
        )

    rows: List[Dict[str, float | int]] = []
    for trial in study.best_trials:
        rows.append(
            {
                "trial_number": trial.number,
                "m": int(trial.params["m"]),
                "ef_construction": int(trial.params["ef_construction"]),
                "ef_search": int(trial.params["ef_search"]),
                "recall": float(trial.values[0]),
                "latency_ms": float(trial.values[1]),
                "build_time_s": float(trial.values[2]),
                "memory_bytes": int(trial.values[3]),
            }
        )

    pareto_df = pd.DataFrame(rows)

    filtered_df = pareto_df[pareto_df["recall"] >= min_recall_for_ranking].copy()
    if not filtered_df.empty:
        pareto_df = filtered_df

    norm_recall = _normalize_benefit(pareto_df["recall"])
    norm_latency = _normalize_cost(pareto_df["latency_ms"])
    norm_build = _normalize_cost(pareto_df["build_time_s"])
    norm_memory = _normalize_cost(pareto_df["memory_bytes"].astype(float))

    pareto_df["aggregate_score"] = (
        recall_weight * norm_recall
        + latency_weight * norm_latency
        + build_time_weight * norm_build
        + memory_weight * norm_memory
    )

    pareto_df = pareto_df.sort_values(
        by=["aggregate_score", "recall", "latency_ms"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    pareto_df.insert(0, "pareto_rank", np.arange(1, len(pareto_df) + 1))
    return pareto_df


def _normalize_benefit(series: pd.Series) -> pd.Series:
    min_v = float(series.min())
    max_v = float(series.max())
    if np.isclose(min_v, max_v):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - min_v) / (max_v - min_v)


def _normalize_cost(series: pd.Series) -> pd.Series:
    min_v = float(series.min())
    max_v = float(series.max())
    if np.isclose(min_v, max_v):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (max_v - series) / (max_v - min_v)
