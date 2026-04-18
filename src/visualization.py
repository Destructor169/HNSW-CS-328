from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def plot_recall_vs_latency(df: pd.DataFrame, output_path: str | Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="latency_ms", y="recall", hue="experiment", ax=ax)
    ax.set_title("Recall vs Latency")
    ax.set_xlabel("Latency per query (ms)")
    ax.set_ylabel("Recall@k")
    _save_figure(fig, output_path)


def plot_recall_vs_parameter(
    df: pd.DataFrame,
    parameter: str,
    output_path: str | Path,
) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(data=df, x=parameter, y="recall", marker="o", ax=ax)
    ax.set_title(f"Recall vs {parameter}")
    ax.set_xlabel(parameter)
    ax.set_ylabel("Recall@k")
    _save_figure(fig, output_path)


def plot_build_time_vs_ef_construction(df: pd.DataFrame, output_path: str | Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(data=df, x="ef_construction", y="build_time_s", marker="o", ax=ax)
    ax.set_title("Build Time vs efConstruction")
    ax.set_xlabel("efConstruction")
    ax.set_ylabel("Build time (s)")
    _save_figure(fig, output_path)


def plot_pareto_frontier(df: pd.DataFrame, output_path: str | Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="latency_ms", y="recall", s=80, ax=ax)

    sorted_df = df.sort_values("latency_ms")
    pareto = []
    best_recall = -1.0
    for _, row in sorted_df.iterrows():
        if row["recall"] > best_recall:
            pareto.append(row)
            best_recall = row["recall"]

    if pareto:
        pareto_df = pd.DataFrame(pareto)
        ax.plot(pareto_df["latency_ms"], pareto_df["recall"], color="red", linewidth=2)

    ax.set_title("Pareto Frontier: Recall vs Latency")
    ax.set_xlabel("Latency per query (ms)")
    ax.set_ylabel("Recall@k")
    _save_figure(fig, output_path)


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
