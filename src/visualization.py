from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_recall_vs_latency(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot recall vs latency with optional Pareto frontier."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    # Handle both latency_ms and latency_p95_ms column names
    latency_col = "latency_ms" if "latency_ms" in df.columns else "latency_p95_ms"
    # Handle both recall and recall_at_10 column names
    recall_col = "recall" if "recall" in df.columns else "recall_at_10"
    
    sns.scatterplot(data=df, x=latency_col, y=recall_col, s=80, ax=ax)
    ax.set_title("Recall vs Latency")
    ax.set_xlabel("Latency per query (ms)")
    ax.set_ylabel("Recall@k")
    _save_figure(fig, output_path)


def plot_pareto_frontier(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot Pareto frontier: recall vs latency with red frontier line."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    # Handle both latency_ms and latency_p95_ms column names
    latency_col = "latency_ms" if "latency_ms" in df.columns else "latency_p95_ms"
    # Handle both recall and recall_at_10 column names
    recall_col = "recall" if "recall" in df.columns else "recall_at_10"
    
    sns.scatterplot(data=df, x=latency_col, y=recall_col, s=80, ax=ax)

    sorted_df = df.sort_values(latency_col)
    pareto = []
    best_recall = -1.0
    for _, row in sorted_df.iterrows():
        if row[recall_col] > best_recall:
            pareto.append(row)
            best_recall = row[recall_col]

    if len(pareto) > 1:  # Only draw line if more than 1 point
        pareto_df = pd.DataFrame(pareto)
        ax.plot(pareto_df[latency_col], pareto_df[recall_col], color="red", linewidth=2, label="Pareto Frontier")
        ax.legend()

    ax.set_title("Pareto Frontier: Recall vs Latency")
    ax.set_xlabel("Latency per query (ms)")
    ax.set_ylabel("Recall@k")
    _save_figure(fig, output_path)


def plot_recall_vs_parameter(
    df: pd.DataFrame,
    parameter: str,
    output_path: str | Path,
) -> None:
    """Plot recall vs a specific parameter."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    # Handle both recall and recall_at_10 column names
    recall_col = "recall" if "recall" in df.columns else "recall_at_10"
    sns.lineplot(data=df, x=parameter, y=recall_col, marker="o", ax=ax)
    ax.set_title(f"Recall vs {parameter}")
    ax.set_xlabel(parameter)
    ax.set_ylabel("Recall@k")
    _save_figure(fig, output_path)


def plot_recall_vs_ef_search(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot recall vs efSearch parameter."""
    # Handle both ef_search and efSearch column names
    ef_search_col = "ef_search" if "ef_search" in df.columns else "efSearch"
    if ef_search_col not in df.columns:
        raise ValueError("Column 'ef_search' or 'efSearch' not found in data")
    
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    recall_col = "recall" if "recall" in df.columns else "recall_at_10"
    
    # Sort by ef_search for better visualization
    plot_df = df.sort_values(ef_search_col)
    sns.scatterplot(data=plot_df, x=ef_search_col, y=recall_col, s=80, ax=ax)
    sns.lineplot(data=plot_df, x=ef_search_col, y=recall_col, ax=ax, legend=False, color="orange", linewidth=2)
    
    ax.set_title("Recall vs efSearch Parameter")
    ax.set_xlabel("efSearch")
    ax.set_ylabel("Recall@k")
    _save_figure(fig, output_path)


def plot_recall_vs_m(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot recall vs M parameter."""
    if "m" not in df.columns:
        raise ValueError("Column 'm' not found in data")
    
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    recall_col = "recall" if "recall" in df.columns else "recall_at_10"
    
    # Sort by m for better visualization
    plot_df = df.sort_values("m")
    sns.scatterplot(data=plot_df, x="m", y=recall_col, s=80, ax=ax)
    sns.lineplot(data=plot_df, x="m", y=recall_col, ax=ax, legend=False, color="green", linewidth=2)
    
    ax.set_title("Recall vs M Parameter")
    ax.set_xlabel("M (Connectivity)")
    ax.set_ylabel("Recall@k")
    _save_figure(fig, output_path)


def plot_build_time_vs_ef_construction(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot build time vs efConstruction parameter."""
    if "ef_construction" not in df.columns:
        raise ValueError("Column 'ef_construction' not found in data")
    
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Sort by ef_construction for better visualization
    plot_df = df.sort_values("ef_construction")
    build_col = "build_time_s" if "build_time_s" in df.columns else "build_time_ms"
    
    sns.scatterplot(data=plot_df, x="ef_construction", y=build_col, s=80, ax=ax)
    sns.lineplot(data=plot_df, x="ef_construction", y=build_col, ax=ax, legend=False, color="purple", linewidth=2)
    
    ax.set_title("Build Time vs efConstruction Parameter")
    ax.set_xlabel("efConstruction")
    ax.set_ylabel("Build time (seconds)")
    _save_figure(fig, output_path)
