from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.dataset_loader import (
    compute_ground_truth,
    prepare_dataset_from_config,
)
from src.experiments import (
    ExperimentConfig,
    compare_default_vs_learned,
    run_parameter_sweeps,
    save_dataframe,
)
from src.optimization import (
    MultiObjectiveConfig,
    OptimizationConfig,
    build_ranked_pareto_report,
    optimize_hnsw_parameters,
    optimize_hnsw_parameters_multi_objective,
)
from src.visualization import (
    plot_build_time_vs_ef_construction,
    plot_pareto_frontier,
    plot_recall_vs_latency,
    plot_recall_vs_parameter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn HNSW construction parameters for ANN search",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve(path_like: str) -> Path:
    return Path(path_like)


def _prepare_dataset(config: Dict[str, Any]) -> tuple[pd.DataFrame, Any, Any]:
    dataset_cfg = config["dataset"]
    prepared = prepare_dataset_from_config(dataset_cfg, seed=int(config["seed"]))
    split = prepared.split

    if prepared.precomputed_ground_truth_indices is not None:
        gt_indices = prepared.precomputed_ground_truth_indices
    else:
        _, gt_indices = compute_ground_truth(
            base_vectors=split.base_vectors,
            query_vectors=split.query_vectors,
            k=int(config["search"]["k"]),
            save_path=_resolve(config["paths"]["ground_truth_indices"]),
        )

    metadata = pd.DataFrame(
        {
            "n_base": [split.base_vectors.shape[0]],
            "n_query": [split.query_vectors.shape[0]],
            "dimension": [split.base_vectors.shape[1]],
            "dataset": [prepared.dataset_label],
        }
    )
    return metadata, split, gt_indices


def _write_report(
    report_path: Path,
    metadata_df: pd.DataFrame,
    learned_params: Dict[str, Any],
    comparison_df: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_info = metadata_df.iloc[0].to_dict()
    default_row = comparison_df.loc[comparison_df["configuration"] == "default"].iloc[0]
    learned_row = comparison_df.loc[comparison_df["configuration"] == "learned"].iloc[0]

    lines = [
        "# Final Report Summary",
        "",
        "## Dataset",
        f"- Source: {dataset_info.get('dataset', 'unknown')}",
        f"- Base vectors: {int(dataset_info['n_base'])}",
        f"- Query vectors: {int(dataset_info['n_query'])}",
        f"- Dimension: {int(dataset_info['dimension'])}",
        "",
        "## Best Learned Parameters",
        f"- M: {learned_params['m']}",
        f"- efConstruction: {learned_params['ef_construction']}",
        f"- efSearch: {learned_params['ef_search']}",
        "",
        "## Performance Comparison",
        "",
        "| configuration | recall | latency_ms | memory_bytes | build_time_s |",
        "|---|---:|---:|---:|---:|",
        (
            f"| default | {default_row['recall']:.4f} | {default_row['latency_ms']:.4f}"
            f" | {int(default_row['memory_bytes'])} | {default_row['build_time_s']:.4f} |"
        ),
        (
            f"| learned | {learned_row['recall']:.4f} | {learned_row['latency_ms']:.4f}"
            f" | {int(learned_row['memory_bytes'])} | {learned_row['build_time_s']:.4f} |"
        ),
        "",
        "## Notes",
        "- Learned parameters are selected by maximizing score = recall - lambda * latency_ms.",
        "- Use results and plots for trade-off analysis and further tuning.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_pareto_report(report_path: Path, pareto_df: pd.DataFrame, top_n: int = 10) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Ranked Pareto Report",
        "",
        "This report lists non-dominated multi-objective trials ranked by a weighted aggregate score.",
        "",
    ]

    if pareto_df.empty:
        lines.append("No Pareto trials were generated.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    top_df = pareto_df.head(top_n)
    lines.extend(
        [
            "## Top Pareto Candidates",
            "",
            "| rank | trial | M | efConstruction | efSearch | recall | latency_ms | build_time_s | memory_bytes | aggregate_score |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in top_df.iterrows():
        lines.append(
            (
                f"| {int(row['pareto_rank'])} | {int(row['trial_number'])} | {int(row['m'])}"
                f" | {int(row['ef_construction'])} | {int(row['ef_search'])} | {row['recall']:.6f}"
                f" | {row['latency_ms']:.6f} | {row['build_time_s']:.6f}"
                f" | {int(row['memory_bytes'])} | {row['aggregate_score']:.6f} |"
            )
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    metadata_df, split, gt_indices = _prepare_dataset(config)

    opt_cfg = config["optimization"]
    optimization_config = OptimizationConfig(
        trials=int(opt_cfg["trials"]),
        k=int(config["search"]["k"]),
        m_min=int(opt_cfg["m_min"]),
        m_max=int(opt_cfg["m_max"]),
        ef_construction_min=int(opt_cfg["ef_construction_min"]),
        ef_construction_max=int(opt_cfg["ef_construction_max"]),
        ef_search_min=int(opt_cfg["ef_search_min"]),
        ef_search_max=int(opt_cfg["ef_search_max"]),
        latency_weight=float(opt_cfg["latency_weight"]),
    )

    study, history = optimize_hnsw_parameters(
        base_vectors=split.base_vectors,
        query_vectors=split.query_vectors,
        ground_truth_indices=gt_indices,
        config=optimization_config,
        seed=int(config["seed"]),
    )

    history_df = pd.DataFrame(history)
    save_dataframe(history_df, _resolve(config["paths"]["optimization_history_csv"]))

    mo_cfg = config.get("multi_objective", {})
    if bool(mo_cfg.get("enabled", True)):
        multi_objective_config = MultiObjectiveConfig(
            enabled=True,
            trials=int(mo_cfg.get("trials", 40)),
            k=int(config["search"]["k"]),
            m_min=int(opt_cfg["m_min"]),
            m_max=int(opt_cfg["m_max"]),
            ef_construction_min=int(opt_cfg["ef_construction_min"]),
            ef_construction_max=int(opt_cfg["ef_construction_max"]),
            ef_search_min=int(opt_cfg["ef_search_min"]),
            ef_search_max=int(opt_cfg["ef_search_max"]),
            score_recall_weight=float(mo_cfg.get("score_recall_weight", 0.5)),
            score_latency_weight=float(mo_cfg.get("score_latency_weight", 0.25)),
            score_build_time_weight=float(mo_cfg.get("score_build_time_weight", 0.15)),
            score_memory_weight=float(mo_cfg.get("score_memory_weight", 0.10)),
            min_recall_for_ranking=float(mo_cfg.get("min_recall_for_ranking", 0.99)),
        )

        mo_study, mo_history = optimize_hnsw_parameters_multi_objective(
            base_vectors=split.base_vectors,
            query_vectors=split.query_vectors,
            ground_truth_indices=gt_indices,
            config=multi_objective_config,
            seed=int(config["seed"]),
        )

        mo_history_df = pd.DataFrame(mo_history)
        save_dataframe(mo_history_df, _resolve(config["paths"]["multi_objective_history_csv"]))

        pareto_df = build_ranked_pareto_report(
            study=mo_study,
            recall_weight=multi_objective_config.score_recall_weight,
            latency_weight=multi_objective_config.score_latency_weight,
            build_time_weight=multi_objective_config.score_build_time_weight,
            memory_weight=multi_objective_config.score_memory_weight,
            min_recall_for_ranking=multi_objective_config.min_recall_for_ranking,
        )
        save_dataframe(pareto_df, _resolve(config["paths"]["pareto_ranked_csv"]))
        _write_pareto_report(_resolve(config["paths"]["pareto_ranked_report_md"]), pareto_df)

    exp_config = ExperimentConfig(
        k=int(config["search"]["k"]),
        default_m=int(config["default_hnsw"]["m"]),
        default_ef_construction=int(config["default_hnsw"]["ef_construction"]),
        default_ef_search=int(config["default_hnsw"]["ef_search"]),
    )
    sweep_df = run_parameter_sweeps(
        base_vectors=split.base_vectors,
        query_vectors=split.query_vectors,
        ground_truth_indices=gt_indices,
        config=exp_config,
    )
    save_dataframe(sweep_df, _resolve(config["paths"]["sweep_results_csv"]))

    best_params = {
        "m": int(study.best_params["m"]),
        "ef_construction": int(study.best_params["ef_construction"]),
        "ef_search": int(study.best_params["ef_search"]),
    }

    comparison_df = compare_default_vs_learned(
        base_vectors=split.base_vectors,
        query_vectors=split.query_vectors,
        ground_truth_indices=gt_indices,
        learned_params=best_params,
        config=exp_config,
    )
    save_dataframe(comparison_df, _resolve(config["paths"]["comparison_csv"]))

    m_df = sweep_df[sweep_df["experiment"] == "effect_of_m"]
    ef_const_df = sweep_df[sweep_df["experiment"] == "effect_of_ef_construction"]
    ef_search_df = sweep_df[sweep_df["experiment"] == "effect_of_ef_search"]

    plot_recall_vs_latency(sweep_df, _resolve(config["paths"]["recall_vs_latency_plot"]))
    plot_recall_vs_parameter(m_df, "m", _resolve(config["paths"]["recall_vs_m_plot"]))
    plot_recall_vs_parameter(
        ef_search_df,
        "ef_search",
        _resolve(config["paths"]["recall_vs_ef_search_plot"]),
    )
    plot_build_time_vs_ef_construction(
        ef_const_df,
        _resolve(config["paths"]["build_time_vs_ef_construction_plot"]),
    )
    plot_pareto_frontier(history_df, _resolve(config["paths"]["pareto_plot"]))

    _write_report(
        report_path=_resolve(config["paths"]["report_md"]),
        metadata_df=metadata_df,
        learned_params=best_params,
        comparison_df=comparison_df,
    )

    print("Optimization complete.")
    print(f"Best params: {best_params}")
    print(f"Best score: {study.best_value:.6f}")
    if bool(mo_cfg.get("enabled", True)):
        print(f"Pareto trials: {len(mo_study.best_trials)}")


if __name__ == "__main__":
    main()
