"""Enhanced CLI for HNSW parameter optimization with multiple strategies."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
import typer
import yaml

from src.dataset_loader import compute_ground_truth, prepare_dataset_from_config
from src.grid_search import grid_search
from src.optimization import (
    MultiObjectiveConfig,
    OptimizationConfig,
    optimize_hnsw_parameters,
    optimize_hnsw_parameters_multi_objective,
)
from src.random_search import random_search
from src.visualization import (
    plot_recall_vs_parameter,
    plot_recall_vs_latency,
    plot_pareto_frontier,
    plot_recall_vs_ef_search,
    plot_recall_vs_m,
    plot_build_time_vs_ef_construction,
)

app = typer.Typer(help="🔍 HNSW Parameter Optimization: Multiple Strategies & Datasets")


def _ensure_output_dir(path: Path) -> Path:
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_timestamp_str() -> str:
    """Get ISO 8601 timestamp string."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _load_config(config_path: str | Path) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _compute_split_hash(dataset_name: str, seed: int) -> str:
    """Compute deterministic hash for dataset splits."""
    key = f"{dataset_name}:{seed}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


@app.command()
def baseline(
    config: str = typer.Option(
        "configs/default.yaml",
        help="Path to YAML config file",
    ),
    ef_search_values: str = typer.Option(
        "20,40,80,160",
        help="Comma-separated efSearch values to sweep",
    ),
    output_dir: str = typer.Option(
        "results/baseline",
        help="Output directory for results",
    ),
    mlflow_tracking: bool = typer.Option(
        False,
        help="Enable MLflow tracking",
    ),
) -> None:
    """Run baseline: fixed (M, efConstruction), sweep efSearch."""
    cfg = _load_config(config)
    output_path = _ensure_output_dir(Path(output_dir))
    
    print("\n" + "="*70)
    print("⚙️  BASELINE: Fixed Parameters, Sweep efSearch")
    print("="*70)
    
    # Prepare dataset
    metadata, split, gt_indices = _prepare_dataset(cfg)
    
    # Parse efSearch values
    ef_search_list = [int(x.strip()) for x in ef_search_values.split(",")]
    
    if mlflow_tracking:
        mlflow.set_experiment("HNSW-Baseline")
        mlflow.start_run(tags={"strategy": "baseline", "dataset": metadata.iloc[0]["dataset"]})
    
    # Get preset model
    model_cfg = cfg.get("models", {}).get("default_hnsw", {})
    m = int(model_cfg.get("m", 16))
    ef_c = int(model_cfg.get("ef_construction", 200))
    k = int(cfg["search"]["k"])
    
    # Run baseline (from optimization.py)
    from src.hnsw_index import HNSWIndexWrapper
    from src.evaluation import summarize_metrics
    import numpy as np
    
    results = []
    print(f"\n📊 Testing with fixed M={m}, efConstruction={ef_c}, varying efSearch={ef_search_list}")
    
    for efs in ef_search_list:
        wrapper = HNSWIndexWrapper(space="l2", dimension=split.base_vectors.shape[1])
        build_start = time.perf_counter()
        build_result = wrapper.build_hnsw_index(
            data=split.base_vectors,
            m=m,
            ef_construction=ef_c,
        )
        build_time = time.perf_counter() - build_start
        
        query_latencies_ms = []
        all_labels = []
        for query in split.query_vectors:
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
        
        metrics = summarize_metrics(
            predicted=predicted_labels,
            ground_truth=gt_indices,
            query_time_seconds=sum(query_latencies_ms) / 1000.0,
            query_latencies_ms=query_latencies_ms,
            num_queries=len(split.query_vectors),
            build_time_seconds=build_time,
            memory_bytes=build_result.index_size_bytes,
            k=k,
        )
        
        score = metrics.recall_at_10 + 0.1 * min(metrics.qps / 1000.0, 1.0)
        
        result = {
            "efSearch": efs,
            "recall_at_10": metrics.recall_at_10,
            "latency_p95_ms": metrics.latency_p95_ms,
            "qps": metrics.qps,
            "build_time_s": metrics.build_time_seconds,
            "score": score,
        }
        results.append(result)
        
        print(f"  efS={efs:3d}: recall@10={metrics.recall_at_10:.4f}, "
              f"p95={metrics.latency_p95_ms:.2f}ms, qps={metrics.qps:.0f}, score={score:.4f}")
        
        if mlflow_tracking:
            mlflow.log_metrics({
                "recall_at_10": metrics.recall_at_10,
                "latency_p95_ms": metrics.latency_p95_ms,
                "qps": metrics.qps,
            })
    
    df = pd.DataFrame(results).sort_values("score", ascending=False)
    csv_path = output_path / f"baseline_{_get_timestamp_str()}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Baseline results saved to {csv_path}")
    
    # Generate plots for baseline (limited since M and efConstruction are fixed)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Baseline only sweeps efSearch, so focus on that
    try:
        plot_recall_vs_ef_search(df, plots_dir / "recall_vs_ef_search.png")
        print(f"✅ Plot: recall_vs_ef_search.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_ef_search: {type(e).__name__}")
    
    try:
        plot_recall_vs_latency(df, plots_dir / "recall_vs_latency.png")
        print(f"✅ Plot: recall_vs_latency.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_latency: {type(e).__name__}")
    
    try:
        plot_pareto_frontier(df, plots_dir / "pareto_frontier.png")
        print(f"✅ Plot: pareto_frontier.png")
    except Exception as e:
        print(f"ℹ️  pareto_frontier: {type(e).__name__}")
    
    # Note: Baseline doesn't have M and efConstruction variation
    print(f"ℹ️  Note: M and efConstruction are fixed in baseline, so recall_vs_m.png and build_time_vs_ef_construction.png not generated")
    
    if mlflow_tracking:
        mlflow.end_run()


@app.command()
def grid(
    config: str = typer.Option(
        "configs/default.yaml",
        help="Path to YAML config file",
    ),
    m_values: str = typer.Option(
        "8,16,32",
        help="Comma-separated M values",
    ),
    ef_construction_values: str = typer.Option(
        "100,200,400",
        help="Comma-separated efConstruction values",
    ),
    ef_search_values: str = typer.Option(
        "20,40,80",
        help="Comma-separated efSearch values",
    ),
    output_dir: str = typer.Option(
        "results/grid_search",
        help="Output directory for results",
    ),
    mlflow_tracking: bool = typer.Option(
        False,
        help="Enable MLflow tracking",
    ),
) -> None:
    """Run grid search: evaluate all parameter combinations."""
    cfg = _load_config(config)
    output_path = _ensure_output_dir(Path(output_dir))
    
    print("\n" + "="*70)
    print("🔲 GRID SEARCH: All Parameter Combinations")
    print("="*70)
    
    # Prepare dataset
    metadata, split, gt_indices = _prepare_dataset(cfg)
    k = int(cfg["search"]["k"])
    
    # Parse parameter values
    m_list = [int(x.strip()) for x in m_values.split(",")]
    efc_list = [int(x.strip()) for x in ef_construction_values.split(",")]
    efs_list = [int(x.strip()) for x in ef_search_values.split(",")]
    
    if mlflow_tracking:
        mlflow.set_experiment("HNSW-GridSearch")
        mlflow.start_run(tags={"strategy": "grid_search", "dataset": metadata.iloc[0]["dataset"]})
    
    # Run grid search
    results = grid_search(
        base_vectors=split.base_vectors,
        query_vectors=split.query_vectors,
        ground_truth_indices=gt_indices,
        m_values=m_list,
        ef_construction_values=efc_list,
        ef_search_values=efs_list,
        k=k,
        output_dir=output_path,
    )
    
    # Generate all 5 standardized plots for grid search
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = results["results_df"]
    
    try:
        plot_recall_vs_latency(df, plots_dir / "recall_vs_latency.png")
        print(f"✅ Plot: recall_vs_latency.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_latency: {type(e).__name__}")
    
    try:
        plot_pareto_frontier(df, plots_dir / "pareto_frontier.png")
        print(f"✅ Plot: pareto_frontier.png")
    except Exception as e:
        print(f"ℹ️  pareto_frontier: {type(e).__name__}")
    
    try:
        plot_recall_vs_ef_search(df, plots_dir / "recall_vs_ef_search.png")
        print(f"✅ Plot: recall_vs_ef_search.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_ef_search: {type(e).__name__}")
    
    try:
        plot_recall_vs_m(df, plots_dir / "recall_vs_m.png")
        print(f"✅ Plot: recall_vs_m.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_m: {type(e).__name__}")
    
    try:
        plot_build_time_vs_ef_construction(df, plots_dir / "build_time_vs_ef_construction.png")
        print(f"✅ Plot: build_time_vs_ef_construction.png")
    except Exception as e:
        print(f"ℹ️  build_time_vs_ef_construction: {type(e).__name__}")
    
    if mlflow_tracking:
        best = results["best_result"]
        mlflow.log_metrics({
            "best_recall_at_10": best["recall_at_10"],
            "best_score": best["score"],
        })
        mlflow.end_run()


@app.command()
def random(
    config: str = typer.Option(
        "configs/default.yaml",
        help="Path to YAML config file",
    ),
    m_min: int = typer.Option(4, help="Minimum M value"),
    m_max: int = typer.Option(64, help="Maximum M value"),
    ef_construction_min: int = typer.Option(50, help="Minimum efConstruction"),
    ef_construction_max: int = typer.Option(500, help="Maximum efConstruction"),
    ef_search_min: int = typer.Option(10, help="Minimum efSearch"),
    ef_search_max: int = typer.Option(300, help="Maximum efSearch"),
    max_trials: int = typer.Option(50, help="Number of random trials"),
    log_scale: bool = typer.Option(True, help="Use log-scaling for ef parameters"),
    output_dir: str = typer.Option(
        "results/random_search",
        help="Output directory for results",
    ),
    mlflow_tracking: bool = typer.Option(
        False,
        help="Enable MLflow tracking",
    ),
) -> None:
    """Run random search: randomly sample parameter combinations."""
    cfg = _load_config(config)
    output_path = _ensure_output_dir(Path(output_dir))
    
    print("\n" + "="*70)
    print("🎲 RANDOM SEARCH: Stochastic Parameter Sampling")
    print("="*70)
    
    # Prepare dataset
    metadata, split, gt_indices = _prepare_dataset(cfg)
    k = int(cfg["search"]["k"])
    
    if mlflow_tracking:
        mlflow.set_experiment("HNSW-RandomSearch")
        mlflow.start_run(tags={"strategy": "random_search", "dataset": metadata.iloc[0]["dataset"]})
    
    # Run random search
    results = random_search(
        base_vectors=split.base_vectors,
        query_vectors=split.query_vectors,
        ground_truth_indices=gt_indices,
        m_min=m_min,
        m_max=m_max,
        ef_construction_min=ef_construction_min,
        ef_construction_max=ef_construction_max,
        ef_search_min=ef_search_min,
        ef_search_max=ef_search_max,
        max_trials=max_trials,
        log_scale_ef=log_scale,
        k=k,
        output_dir=output_path,
    )
    
    # Generate all 5 standardized plots for random search
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = results["results_df"]
    
    try:
        plot_recall_vs_latency(df, plots_dir / "recall_vs_latency.png")
        print(f"✅ Plot: recall_vs_latency.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_latency: {type(e).__name__}")
    
    try:
        plot_pareto_frontier(df, plots_dir / "pareto_frontier.png")
        print(f"✅ Plot: pareto_frontier.png")
    except Exception as e:
        print(f"ℹ️  pareto_frontier: {type(e).__name__}")
    
    try:
        plot_recall_vs_ef_search(df, plots_dir / "recall_vs_ef_search.png")
        print(f"✅ Plot: recall_vs_ef_search.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_ef_search: {type(e).__name__}")
    
    try:
        plot_recall_vs_m(df, plots_dir / "recall_vs_m.png")
        print(f"✅ Plot: recall_vs_m.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_m: {type(e).__name__}")
    
    try:
        plot_build_time_vs_ef_construction(df, plots_dir / "build_time_vs_ef_construction.png")
        print(f"✅ Plot: build_time_vs_ef_construction.png")
    except Exception as e:
        print(f"ℹ️  build_time_vs_ef_construction: {type(e).__name__}")
    
    if mlflow_tracking:
        best = results["best_result"]
        mlflow.log_metrics({
            "best_recall_at_10": best["recall_at_10"],
            "best_score": best["score"],
        })
        mlflow.end_run()


@app.command()
def bayesian(
    config: str = typer.Option(
        "configs/default.yaml",
        help="Path to YAML config file",
    ),
    m_min: int = typer.Option(4, help="Minimum M value"),
    m_max: int = typer.Option(64, help="Maximum M value"),
    ef_construction_min: int = typer.Option(50, help="Minimum efConstruction"),
    ef_construction_max: int = typer.Option(500, help="Maximum efConstruction"),
    ef_search_min: int = typer.Option(10, help="Minimum efSearch"),
    ef_search_max: int = typer.Option(300, help="Maximum efSearch"),
    max_trials: int = typer.Option(100, help="Number of optimization trials"),
    startup_trials: int = typer.Option(10, help="Number of random startup trials"),
    latency_weight: float = typer.Option(0.001, help="Latency penalty weight"),
    output_dir: str = typer.Option(
        "results/bayesian",
        help="Output directory for results",
    ),
    mlflow_tracking: bool = typer.Option(
        False,
        help="Enable MLflow tracking",
    ),
) -> None:
    """Run Bayesian optimization (Optuna TPE)."""
    cfg = _load_config(config)
    output_path = _ensure_output_dir(Path(output_dir))
    
    print("\n" + "="*70)
    print("🧠 BAYESIAN OPTIMIZATION: Optuna TPE Sampler")
    print("="*70)
    
    # Prepare dataset
    metadata, split, gt_indices = _prepare_dataset(cfg)
    k = int(cfg["search"]["k"])
    
    opt_cfg = OptimizationConfig(
        trials=max_trials,
        k=k,
        m_min=m_min,
        m_max=m_max,
        ef_construction_min=ef_construction_min,
        ef_construction_max=ef_construction_max,
        ef_search_min=ef_search_min,
        ef_search_max=ef_search_max,
        latency_weight=latency_weight,
    )
    
    if mlflow_tracking:
        mlflow.set_experiment("HNSW-BayesianOptimization")
        mlflow.start_run(tags={"strategy": "bayesian", "dataset": metadata.iloc[0]["dataset"]})
    
    # Run optimization
    results = optimize_hnsw_parameters(
        base_vectors=split.base_vectors,
        query_vectors=split.query_vectors,
        ground_truth_indices=gt_indices,
        config=opt_cfg,
        n_startup_trials=startup_trials,
        output_dir=output_path,
    )
    
    # Generate plots
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = results["results_df"]
    
    try:
        plot_recall_vs_latency(df, plots_dir / "recall_vs_latency.png")
        print(f"✅ Plot: recall_vs_latency.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_latency: {type(e).__name__}")
    
    try:
        plot_pareto_frontier(df, plots_dir / "pareto_frontier.png")
        print(f"✅ Plot: pareto_frontier.png")
    except Exception as e:
        print(f"ℹ️  pareto_frontier: {type(e).__name__}")
    
    try:
        plot_recall_vs_ef_search(df, plots_dir / "recall_vs_ef_search.png")
        print(f"✅ Plot: recall_vs_ef_search.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_ef_search: {type(e).__name__}")
    
    try:
        plot_recall_vs_m(df, plots_dir / "recall_vs_m.png")
        print(f"✅ Plot: recall_vs_m.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_m: {type(e).__name__}")
    
    try:
        plot_build_time_vs_ef_construction(df, plots_dir / "build_time_vs_ef_construction.png")
        print(f"✅ Plot: build_time_vs_ef_construction.png")
    except Exception as e:
        print(f"ℹ️  build_time_vs_ef_construction: {type(e).__name__}")
    
    if mlflow_tracking:
        mlflow.log_metrics({
            "best_recall_at_10": results["best_result"]["recall_at_10"],
            "best_score": results["best_result"]["score"],
        })
        mlflow.end_run()


@app.command()
def multi_objective(
    config: str = typer.Option(
        "configs/default.yaml",
        help="Path to YAML config file",
    ),
    max_trials: int = typer.Option(100, help="Number of trials"),
    min_recall: float = typer.Option(0.99, help="Minimum recall filter for Pareto"),
    recall_weight: float = typer.Option(0.5, help="Weight for recall in scoring"),
    latency_weight: float = typer.Option(0.25, help="Weight for latency"),
    build_time_weight: float = typer.Option(0.15, help="Weight for build time"),
    memory_weight: float = typer.Option(0.10, help="Weight for memory"),
    output_dir: str = typer.Option(
        "results/multi_objective",
        help="Output directory for results",
    ),
    mlflow_tracking: bool = typer.Option(
        False,
        help="Enable MLflow tracking",
    ),
) -> None:
    """Run multi-objective optimization (4D Pareto frontier)."""
    cfg = _load_config(config)
    output_path = _ensure_output_dir(Path(output_dir))
    
    print("\n" + "="*70)
    print("🎯 MULTI-OBJECTIVE OPTIMIZATION: 4D Pareto Frontier")
    print("="*70)
    
    # Prepare dataset
    metadata, split, gt_indices = _prepare_dataset(cfg)
    k = int(cfg["search"]["k"])
    
    mo_cfg = MultiObjectiveConfig(
        enabled=True,
        trials=max_trials,
        k=k,
        score_recall_weight=recall_weight,
        score_latency_weight=latency_weight,
        score_build_time_weight=build_time_weight,
        score_memory_weight=memory_weight,
        min_recall_for_ranking=min_recall,
    )
    
    if mlflow_tracking:
        mlflow.set_experiment("HNSW-MultiObjective")
        mlflow.start_run(tags={"strategy": "multi_objective", "dataset": metadata.iloc[0]["dataset"]})
    
    # Run multi-objective optimization
    results = optimize_hnsw_parameters_multi_objective(
        base_vectors=split.base_vectors,
        query_vectors=split.query_vectors,
        ground_truth_indices=gt_indices,
        config=mo_cfg,
        output_dir=output_path,
    )
    
    if mlflow_tracking:
        mlflow.log_metrics({
            "pareto_front_size": len(results["pareto_df"]),
            "best_recall_at_10": results["best_result"]["recall_at_10"],
        })
        mlflow.end_run()
    
    # Generate all 5 standardized plots for multi-objective
    pareto_df = results["pareto_df"]
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        plot_recall_vs_latency(pareto_df, plots_dir / "recall_vs_latency.png")
        print(f"✅ Plot: recall_vs_latency.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_latency: {type(e).__name__}")
    
    try:
        plot_pareto_frontier(pareto_df, plots_dir / "pareto_frontier.png")
        print(f"✅ Plot: pareto_frontier.png")
    except Exception as e:
        print(f"ℹ️  pareto_frontier: {type(e).__name__}")
    
    try:
        plot_recall_vs_ef_search(pareto_df, plots_dir / "recall_vs_ef_search.png")
        print(f"✅ Plot: recall_vs_ef_search.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_ef_search: {type(e).__name__}")
    
    try:
        plot_recall_vs_m(pareto_df, plots_dir / "recall_vs_m.png")
        print(f"✅ Plot: recall_vs_m.png")
    except Exception as e:
        print(f"ℹ️  recall_vs_m: {type(e).__name__}")
    
    try:
        plot_build_time_vs_ef_construction(pareto_df, plots_dir / "build_time_vs_ef_construction.png")
        print(f"✅ Plot: build_time_vs_ef_construction.png")
    except Exception as e:
        print(f"ℹ️  build_time_vs_ef_construction: {type(e).__name__}")
    
    print(f"\n✅ Multi-objective optimization complete!")
    print(f"   📊 Pareto front size: {len(pareto_df)}")
    print(f"   💾 Results saved to: {output_path / 'multi_objective_pareto.csv'}")
    print(f"   📈 5 plots generated in: {plots_dir}")


def _prepare_dataset(cfg: dict) -> tuple:
    """Prepare dataset split and ground truth."""
    dataset_cfg = cfg["dataset"]
    seed = int(cfg.get("seed", 42))
    
    prepared = prepare_dataset_from_config(dataset_cfg, seed=seed)
    split = prepared.split
    
    k = int(cfg["search"]["k"])
    _, gt_indices = compute_ground_truth(
        base_vectors=split.base_vectors,
        query_vectors=split.query_vectors,
        k=k,
        metric="euclidean",
    )
    
    metadata = pd.DataFrame({
        "n_base": [split.base_vectors.shape[0]],
        "n_query": [split.query_vectors.shape[0]],
        "dimension": [split.base_vectors.shape[1]],
        "dataset": [prepared.dataset_label],
    })
    
    return metadata, split, gt_indices


if __name__ == "__main__":
    app()
