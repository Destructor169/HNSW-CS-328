from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.dataset_loader import generate_synthetic_dataset, normalize_vectors, split_dataset


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results"
DOC_FILES = {
    "README": REPO_ROOT / "README.md",
    "Architecture": REPO_ROOT / "ARCHITECTURE.md",
    "Results": REPO_ROOT / "RESULTS.md",
}


st.set_page_config(
    page_title="HNSW-CS-328 Interactive Report",
    page_icon="📊",
    layout="wide",
)


def apply_professional_theme() -> None:
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 1.5rem;
            }
            .intro-card {
                background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 1rem 1.1rem;
                margin-bottom: 0.8rem;
                color: #E2E8F0;
            }
            .intro-title {
                color: #38BDF8;
                font-weight: 700;
                margin-bottom: 0.4rem;
            }
            .intro-muted {
                color: #94A3B8;
                font-size: 0.94rem;
            }
            .kpi-chip {
                display: inline-block;
                background: #0EA5E9;
                color: white;
                border-radius: 999px;
                padding: 0.2rem 0.6rem;
                font-size: 0.78rem;
                margin-right: 0.35rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def read_markdown(path: Path) -> str:
    if not path.exists():
        return f"⚠️ Missing file: {path}"
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def discover_csv_files(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(base_dir.rglob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def discover_plot_files(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(base_dir.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def pick_best_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None

    for key in ("score", "recall_at_10", "recall"):
        if key in df.columns:
            return df.loc[df[key].idxmax()]

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return None
    return df.iloc[0]


def summary_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    out: dict[str, float | int] = {"rows": int(len(df))}

    if "recall_at_10" in df.columns:
        out["best_recall_at_10"] = float(df["recall_at_10"].max())
    elif "recall" in df.columns:
        out["best_recall"] = float(df["recall"].max())

    if "latency_p95_ms" in df.columns:
        out["best_latency_p95_ms"] = float(df["latency_p95_ms"].min())
    elif "latency_ms" in df.columns:
        out["best_latency_ms"] = float(df["latency_ms"].min())

    if "qps" in df.columns:
        out["best_qps"] = float(df["qps"].max())

    if "score" in df.columns:
        out["best_score"] = float(df["score"].max())

    return out


def run_cli_command(cmd: list[str]) -> tuple[int, str]:
    process = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    logs = (process.stdout or "") + "\n" + (process.stderr or "")
    return process.returncode, logs.strip()


@st.cache_data(show_spinner=False)
def build_projection_from_base_query(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    projection: str,
    seed: int,
    max_base_points: int,
    max_query_points: int,
    tsne_perplexity: int,
    tsne_iterations: int,
    source_label: str,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    base = base_vectors
    query = query_vectors

    if len(base) > max_base_points:
        base_idx = rng.choice(len(base), size=max_base_points, replace=False)
        base = base[base_idx]

    if len(query) > max_query_points:
        query_idx = rng.choice(len(query), size=max_query_points, replace=False)
        query = query[query_idx]

    stacked = np.vstack([base, query]).astype(np.float32, copy=False)
    labels = np.array(["base"] * len(base) + ["query"] * len(query))

    if projection.lower() == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        coords = reducer.fit_transform(stacked)
        meta = {
            "projection": "PCA",
            "explained_var_pc1": float(reducer.explained_variance_ratio_[0]),
            "explained_var_pc2": float(reducer.explained_variance_ratio_[1]),
        }
    else:
        reducer = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            max_iter=tsne_iterations,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        )
        coords = reducer.fit_transform(stacked)
        meta = {
            "projection": "t-SNE",
            "perplexity": tsne_perplexity,
            "iterations": tsne_iterations,
        }

    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "set": labels})
    meta.update(
        {
            "source": source_label,
            "n_base_shown": int((df["set"] == "base").sum()),
            "n_query_shown": int((df["set"] == "query").sum()),
            "n_base_total": int(base_vectors.shape[0]),
            "n_query_total": int(query_vectors.shape[0]),
            "dimension": int(base_vectors.shape[1]),
        }
    )
    if extra_meta:
        meta.update(extra_meta)
    return df, meta


@st.cache_data(show_spinner=False)
def build_synthetic_projection(
    n_vectors: int,
    dimension: int,
    n_clusters: int,
    cluster_std: float,
    query_fraction: float,
    normalize: bool,
    seed: int,
    projection: str,
    max_base_points: int,
    max_query_points: int,
    tsne_perplexity: int,
    tsne_iterations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    vectors = generate_synthetic_dataset(
        n_vectors=n_vectors,
        dimension=dimension,
        n_clusters=n_clusters,
        cluster_std=cluster_std,
        random_state=seed,
    )
    if normalize:
        vectors = normalize_vectors(vectors)
    split = split_dataset(vectors=vectors, query_fraction=query_fraction, random_state=seed)
    return build_projection_from_base_query(
        base_vectors=split.base_vectors,
        query_vectors=split.query_vectors,
        projection=projection,
        seed=seed,
        max_base_points=max_base_points,
        max_query_points=max_query_points,
        tsne_perplexity=tsne_perplexity,
        tsne_iterations=tsne_iterations,
        source_label="synthetic",
        extra_meta={
            "n_vectors_total": int(n_vectors),
            "n_clusters": int(n_clusters),
            "cluster_std": float(cluster_std),
        },
    )


@st.cache_data(show_spinner=False)
def load_real_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    dataset_dir = REPO_ROOT / "data" / "benchmarks" / dataset_name
    base_path = dataset_dir / "base.npy"
    query_path = dataset_dir / "query.npy"

    if not base_path.exists() or not query_path.exists():
        raise FileNotFoundError(f"Missing files for dataset '{dataset_name}' in {dataset_dir}")

    base = np.load(base_path).astype(np.float32, copy=False)
    query = np.load(query_path).astype(np.float32, copy=False)
    return base, query


@st.cache_data(show_spinner=False)
def build_real_projection(
    dataset_name: str,
    projection: str,
    seed: int,
    max_base_points: int,
    max_query_points: int,
    tsne_perplexity: int,
    tsne_iterations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base, query = load_real_dataset(dataset_name)
    return build_projection_from_base_query(
        base_vectors=base,
        query_vectors=query,
        projection=projection,
        seed=seed,
        max_base_points=max_base_points,
        max_query_points=max_query_points,
        tsne_perplexity=tsne_perplexity,
        tsne_iterations=tsne_iterations,
        source_label=dataset_name,
    )


def _extract_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    default_hnsw = cfg.get("default_hnsw", {})
    # Backward compatibility with older config style
    if not default_hnsw:
        default_hnsw = cfg.get("models", {}).get("default_hnsw", {})

    optimization = cfg.get("optimization", {})
    multi_objective = cfg.get("multi_objective", {})

    return {
        "m": default_hnsw.get("m", "n/a"),
        "ef_construction": default_hnsw.get("ef_construction", "n/a"),
        "ef_search": default_hnsw.get("ef_search", "n/a"),
        "k": cfg.get("search", {}).get("k", "n/a"),
        "trials": optimization.get("trials", "n/a"),
        "m_range": f"{optimization.get('m_min', 'n/a')} - {optimization.get('m_max', 'n/a')}",
        "ef_construction_range": f"{optimization.get('ef_construction_min', 'n/a')} - {optimization.get('ef_construction_max', 'n/a')}",
        "ef_search_range": f"{optimization.get('ef_search_min', 'n/a')} - {optimization.get('ef_search_max', 'n/a')}",
        "latency_weight": optimization.get("latency_weight", "n/a"),
        "mo_trials": multi_objective.get("trials", "n/a"),
    }


def _best_tuned_parameters() -> dict[str, Any]:
    csv_files = discover_csv_files(RESULTS_ROOT)
    for path in csv_files:
        try:
            df = load_csv(path)
        except Exception:
            continue

        best = pick_best_row(df)
        if best is None:
            continue

        keys = ["m", "ef_construction", "ef_search", "recall_at_10", "latency_p95_ms", "qps", "score", "recall", "latency_ms"]
        payload = {k: best.get(k) for k in keys if k in best.index}
        if any(k in payload for k in ("m", "ef_construction", "ef_search")):
            payload["source_csv"] = relative(path)
            return payload

    return {}


def render_introduction_page() -> None:
    st.subheader("Introduction")
    st.markdown(
        """
        <div class="intro-card">
            <div class="intro-title">HNSW + ANN at Scale</div>
            <div>
                This project builds an end-to-end optimization framework for <b>ANN (Approximate Nearest Neighbor)</b>
                search using <b>HNSW (Hierarchical Navigable Small World)</b> indexes. ANN is used when exact KNN is
                too expensive at production scale, and HNSW is one of the most practical methods for balancing
                <b>quality</b>, <b>latency</b>, and <b>memory</b>.
            </div>
            <div style="margin-top:0.6rem;">
                <span class="kpi-chip">Vector Search</span>
                <span class="kpi-chip">Recommendations</span>
                <span class="kpi-chip">Image Similarity</span>
                <span class="kpi-chip">Fraud / Entity Matching</span>
                <span class="kpi-chip">RAG Retrieval</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project Structure & Workflow")
    workflow_steps = [
        "Data preparation (synthetic / benchmark datasets)",
        "Ground-truth generation (exact KNN)",
        "Hyperparameter strategy execution",
        "Evaluation (recall, latency, throughput, build-time, memory)",
        "Reporting (CSV + plots + markdown summaries)",
    ]
    selected_step = st.select_slider(
        "Workflow navigator",
        options=workflow_steps,
        value=workflow_steps[0],
    )
    st.info(f"Current stage: **{selected_step}**")

    st.markdown(
        """
        <div class="intro-muted">
        Strategy focus: move from a controlled baseline to broader and smarter exploration, then rank candidates by
        business trade-offs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Parameter Definitions (Broad)")
    st.markdown(
        """
- **`M`**: graph connectivity; larger values typically improve recall but increase memory/build cost.
- **`efConstruction`**: index build effort; higher values usually improve index quality but slow build.
- **`efSearch`**: query-time exploration budget; higher values generally improve recall but increase latency.
- **`k`**: number of neighbors returned/evaluated.
"""
    )

    st.markdown("### Metrics (Broad)")
    st.markdown(
        """
- **Recall@k**: quality/accuracy of nearest-neighbor retrieval.
- **Latency (p50/p95/p99)**: response-time behavior, especially tail latency.
- **QPS**: throughput (queries per second).
- **Build time**: indexing cost.
- **Memory**: runtime footprint of the index.
"""
    )

    st.markdown("### Hyperparameter Tuning Strategies")
    strategy_notes = {
        "Baseline": "Controlled one-variable sweep (typically efSearch) with fixed index topology to establish a reference curve.",
        "Grid Search": "Exhaustive search across a bounded parameter grid; high coverage, high cost.",
        "Random Search": "Stochastic exploration over wide ranges; often strong results with fewer trials than grid.",
        "Bayesian Optimization": "Model-guided sampling using prior trial outcomes; efficient when evaluations are expensive.",
        "Multi-Objective": "Optimizes multiple goals jointly (recall, latency, build-time, memory) and returns a Pareto frontier.",
    }
    strategy = st.radio("Select a strategy", list(strategy_notes.keys()), horizontal=True)
    st.markdown(
        f"""
        <div class="intro-card">
            <div class="intro-title">{strategy}</div>
            <div>{strategy_notes[strategy]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Interactive trade-off intuition")
    c1, c2, c3 = st.columns(3)
    m_level = c1.slider("M emphasis", 1, 10, 6)
    efc_level = c2.slider("efConstruction emphasis", 1, 10, 5)
    efs_level = c3.slider("efSearch emphasis", 1, 10, 7)

    expected_recall = min(100, 45 + m_level * 2 + efs_level * 2)
    expected_latency = max(1, 5 + efs_level * 2 + (m_level // 2))
    expected_build = max(1, 4 + efc_level * 2 + (m_level // 2))

    t1, t2, t3 = st.columns(3)
    t1.metric("Estimated Recall Trend", f"{expected_recall}%")
    t2.metric("Estimated Query Cost Trend", f"{expected_latency}/10")
    t3.metric("Estimated Build Cost Trend", f"{expected_build}/10")

    cfg_path = REPO_ROOT / "configs" / "default.yaml"
    if cfg_path.exists():
        cfg = load_config(cfg_path)
        defaults = _extract_defaults(cfg)

        st.markdown("### Default Values (from config)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Default M", defaults["m"])
        c2.metric("Default efConstruction", defaults["ef_construction"])
        c3.metric("Default efSearch", defaults["ef_search"])
        c4.metric("Top-k", defaults["k"])

        st.markdown("#### Optimization Ranges")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("M range", defaults["m_range"])
        r2.metric("efConstruction range", defaults["ef_construction_range"])
        r3.metric("efSearch range", defaults["ef_search_range"])
        r4.metric("Trials", defaults["trials"])

    updated = _best_tuned_parameters()
    st.markdown("### Updated / Best Observed Parameters")
    if not updated:
        st.info("No tuned result CSV found yet. Run a strategy from the HyperTune tab to populate this section.")
    else:
        u1, u2, u3, u4 = st.columns(4)
        if "m" in updated:
            u1.metric("Best M", updated["m"])
        if "ef_construction" in updated:
            u2.metric("Best efConstruction", updated["ef_construction"])
        if "ef_search" in updated:
            u3.metric("Best efSearch", updated["ef_search"])
        if "recall_at_10" in updated:
            u4.metric("Best Recall@10", f"{float(updated['recall_at_10']):.4f}")
        elif "recall" in updated:
            u4.metric("Best Recall", f"{float(updated['recall']):.4f}")

        if "source_csv" in updated:
            st.caption(f"Source: {updated['source_csv']}")


def render_run_outputs(output_dir: Path) -> None:
    st.markdown("### Generated Outputs")
    if not output_dir.exists():
        st.info(f"No output directory found at `{relative(output_dir)}`")
        return

    csv_files = discover_csv_files(output_dir)
    png_files = discover_plot_files(output_dir)

    if csv_files:
        latest_csv = csv_files[0]
        st.write(f"Latest CSV: `{relative(latest_csv)}`")
        df = load_csv(latest_csv)
        st.dataframe(df, use_container_width=True)

    if png_files:
        st.markdown("#### Plots")
        for idx in range(0, len(png_files), 2):
            cols = st.columns(2)
            left = png_files[idx]
            cols[0].image(str(left), caption=relative(left), use_container_width=True)
            if idx + 1 < len(png_files):
                right = png_files[idx + 1]
                cols[1].image(str(right), caption=relative(right), use_container_width=True)
    elif not csv_files:
        st.info("Run completed, but no CSV/PNG artifacts were found in the output path.")


def render_project_summary() -> None:
    st.subheader("Project Summary")

    cfg_path = REPO_ROOT / "configs" / "default.yaml"
    if cfg_path.exists():
        cfg = load_config(cfg_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset Source", cfg.get("dataset", {}).get("source", "n/a"))
        c2.metric("Top-k", cfg.get("search", {}).get("k", "n/a"))
        c3.metric("Seed", cfg.get("seed", "n/a"))
        c4.metric("Optimization Trials", cfg.get("optimization", {}).get("trials", "n/a"))

    st.markdown("This dashboard combines:")
    st.markdown("- **Report view** of markdown docs (`README`, `ARCHITECTURE`, `RESULTS`)")
    st.markdown("- **Interactive results explorer** for CSV/plot artifacts")
    st.markdown("- **Hyperparameter tuning runner** powered by your existing Typer CLIs")


def render_synthetic_visualizer() -> None:
    st.subheader("Dataset Visualizer")
    st.caption("Visualize synthetic or real datasets (SIFT-1M / GloVe-100) with PCA/t-SNE and base/query overlays.")

    dataset_choice = st.selectbox("Dataset", ["synthetic", "sift1m", "glove100"], index=0)

    cfg_path = REPO_ROOT / "configs" / "default.yaml"
    cfg = load_config(cfg_path) if cfg_path.exists() else {}
    synth_cfg = cfg.get("dataset", {}).get("synthetic", {})
    seed_default = int(cfg.get("seed", 42))
    normalize_default = bool(cfg.get("dataset", {}).get("normalize", True))
    query_fraction_default = float(cfg.get("dataset", {}).get("query_fraction", 0.1))

    # Synthetic controls only
    if dataset_choice == "synthetic":
        c1, c2, c3 = st.columns(3)
        n_vectors = c1.slider("n_vectors", min_value=2000, max_value=50000, value=int(synth_cfg.get("n_vectors", 20000)), step=1000)
        dimension = c2.slider("dimension", min_value=8, max_value=512, value=int(synth_cfg.get("dimension", 128)), step=8)
        n_clusters = c3.slider("n_clusters", min_value=2, max_value=200, value=int(synth_cfg.get("n_clusters", 40)), step=1)

        c4, c5, c6 = st.columns(3)
        cluster_std = c4.slider("cluster_std", min_value=0.1, max_value=5.0, value=float(synth_cfg.get("cluster_std", 1.5)), step=0.1)
        query_fraction = c5.slider("query_fraction", min_value=0.05, max_value=0.5, value=float(query_fraction_default), step=0.01)
        normalize = c6.toggle("L2 normalize", value=normalize_default)
    else:
        st.info(
            f"Selected real dataset: **{dataset_choice}**. "
            "Use the point sliders below to control projection size for faster visualization."
        )

    c7, c8, c9 = st.columns(3)
    projection = c7.selectbox("Projection", options=["PCA", "t-SNE"], index=0)
    max_base_points = c8.slider("max base points shown", min_value=500, max_value=50000, value=5000, step=500)
    max_query_points = c9.slider("max query points shown", min_value=100, max_value=50000, value=1500, step=100)

    c10, c11 = st.columns(2)
    tsne_perplexity = c10.slider("t-SNE perplexity", min_value=5, max_value=80, value=30, step=1)
    tsne_iterations = c11.slider("t-SNE iterations", min_value=250, max_value=2000, value=1000, step=50)

    seed = st.number_input("random seed", min_value=0, max_value=10_000, value=seed_default, step=1)

    with st.spinner("Preparing and projecting vectors..."):
        if dataset_choice == "synthetic":
            df, meta = build_synthetic_projection(
                n_vectors=n_vectors,
                dimension=dimension,
                n_clusters=n_clusters,
                cluster_std=cluster_std,
                query_fraction=query_fraction,
                normalize=normalize,
                seed=int(seed),
                projection=projection,
                max_base_points=max_base_points,
                max_query_points=max_query_points,
                tsne_perplexity=tsne_perplexity,
                tsne_iterations=tsne_iterations,
            )
        else:
            try:
                df, meta = build_real_projection(
                    dataset_name=dataset_choice,
                    projection=projection,
                    seed=int(seed),
                    max_base_points=max_base_points,
                    max_query_points=max_query_points,
                    tsne_perplexity=tsne_perplexity,
                    tsne_iterations=tsne_iterations,
                )
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset", meta.get("source", "n/a"))
    m2.metric("Projection", meta.get("projection", "n/a"))
    m3.metric("Base points shown", meta.get("n_base_shown", 0))
    m4.metric("Query points shown", meta.get("n_query_shown", 0))

    cmeta1, cmeta2, cmeta3 = st.columns(3)
    cmeta1.metric("Total base vectors", meta.get("n_base_total", 0))
    cmeta2.metric("Total query vectors", meta.get("n_query_total", 0))
    cmeta3.metric("Dimension", meta.get("dimension", 0))

    if meta.get("projection") == "PCA":
        st.caption(
            f"PCA explained variance: PC1={meta.get('explained_var_pc1', 0):.2%}, "
            f"PC2={meta.get('explained_var_pc2', 0):.2%}"
        )
    elif meta.get("projection") == "t-SNE":
        st.caption(
            f"t-SNE settings: perplexity={meta.get('perplexity')}, iterations={meta.get('iterations')}"
        )

    if dataset_choice == "synthetic":
        st.caption(
            f"Synthetic generation: n_vectors={meta.get('n_vectors_total')}, "
            f"n_clusters={meta.get('n_clusters')}, cluster_std={meta.get('cluster_std')}"
        )

    base_df = df[df["set"] == "base"]
    query_df = df[df["set"] == "query"]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(base_df["x"], base_df["y"], s=20, alpha=0.55, c="#1E40AF", label="Base", marker="o")
    ax.scatter(query_df["x"], query_df["y"], s=36, alpha=0.85, c="#EA580C", label="Query", marker="^")
    ax.set_xlabel(f"{projection} component 1")
    ax.set_ylabel(f"{projection} component 2")
    ax.set_title("Synthetic embedding with base/query overlays")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)

    with st.expander("Sample projected points"):
        st.dataframe(df.head(200), use_container_width=True)


def render_results_explorer() -> None:
    st.subheader("Results Explorer")

    base = st.selectbox(
        "Results root",
        options=[str(RESULTS_ROOT / "dataset_results"), str(RESULTS_ROOT / "final_results"), str(RESULTS_ROOT)],
        index=0,
    )
    base_path = Path(base)

    csv_files = discover_csv_files(base_path)
    if not csv_files:
        st.warning(f"No CSV results found under {relative(base_path)}")
        return

    selected_csv = st.selectbox("Select CSV", options=[relative(p) for p in csv_files])
    csv_path = REPO_ROOT / selected_csv

    df = load_csv(csv_path)
    st.caption(f"Loaded {relative(csv_path)}")

    metrics = summary_metrics(df)
    cols = st.columns(min(5, max(1, len(metrics))))
    for i, (k, v) in enumerate(metrics.items()):
        cols[i % len(cols)].metric(k, f"{v:.4f}" if isinstance(v, float) else v)

    best = pick_best_row(df)
    if best is not None:
        with st.expander("Best configuration row", expanded=True):
            st.json({k: (None if pd.isna(v) else v) for k, v in best.to_dict().items()})

    st.dataframe(df, use_container_width=True)

    st.markdown("### Interactive Charts")
    numeric_cols = list(df.select_dtypes(include="number").columns)

    if len(numeric_cols) >= 2:
        c1, c2 = st.columns(2)
        x_axis = c1.selectbox("X-axis", options=numeric_cols, index=0)
        y_default_idx = 1 if len(numeric_cols) > 1 else 0
        y_axis = c2.selectbox("Y-axis", options=numeric_cols, index=y_default_idx)

        chart_df = df[[x_axis, y_axis]].dropna()
        st.scatter_chart(chart_df, x=x_axis, y=y_axis, use_container_width=True)

    # Common project charts
    if {"latency_p95_ms", "recall_at_10"}.issubset(df.columns):
        st.markdown("#### Recall@10 vs Latency P95")
        st.scatter_chart(df[["latency_p95_ms", "recall_at_10"]], x="latency_p95_ms", y="recall_at_10", use_container_width=True)

    if {"ef_search", "recall_at_10"}.issubset(df.columns):
        st.markdown("#### Recall@10 vs ef_search")
        line_df = df.sort_values("ef_search")[["ef_search", "recall_at_10"]]
        st.line_chart(line_df.set_index("ef_search"), use_container_width=True)

    if {"m", "recall_at_10"}.issubset(df.columns):
        st.markdown("#### Recall@10 vs M")
        line_df = df.sort_values("m")[["m", "recall_at_10"]]
        st.line_chart(line_df.set_index("m"), use_container_width=True)

    st.markdown("### Plot Gallery")
    png_files = discover_plot_files(base_path)
    if not png_files:
        st.info("No PNG plots found in selected result root.")
        return

    selected_png = st.selectbox("Select plot", options=[relative(p) for p in png_files])
    st.image(str(REPO_ROOT / selected_png), caption=selected_png, use_container_width=True)


def render_docs_panel() -> None:
    st.subheader("Documentation")
    doc_choice = st.radio("Document", options=list(DOC_FILES.keys()), horizontal=True)
    st.markdown(read_markdown(DOC_FILES[doc_choice]))


def render_tuning_runner() -> None:
    st.subheader("Hyperparameter Tuning Runner")
    st.caption("This section launches your existing CLI commands and writes outputs to `results/`.")

    mode = st.selectbox("CLI mode", ["src.cli (general)", "src.dataset_cli (real datasets)"])

    with st.form("run_form"):
        if mode == "src.cli (general)":
            strategy = st.selectbox("Strategy", ["baseline", "grid", "random", "bayesian", "multi-objective"])
            config = st.text_input("Config path", "configs/default.yaml")
            output_dir = st.text_input("Output dir", f"results/interactive/{strategy}")

            cmd = [sys.executable, "-m", "src.cli", strategy, "--config", config, "--output-dir", output_dir]

            if strategy == "baseline":
                ef_vals = st.text_input("ef-search-values", "20,40,80,160")
                cmd += ["--ef-search-values", ef_vals]
            elif strategy == "grid":
                m_vals = st.text_input("m-values", "8,16,32")
                efc_vals = st.text_input("ef-construction-values", "100,200,400")
                efs_vals = st.text_input("ef-search-values", "20,40,80")
                cmd += ["--m-values", m_vals, "--ef-construction-values", efc_vals, "--ef-search-values", efs_vals]
            elif strategy == "random":
                col1, col2 = st.columns(2)
                m_min = col1.number_input("m-min", min_value=2, max_value=128, value=4)
                m_max = col2.number_input("m-max", min_value=2, max_value=256, value=64)
                efc_min = col1.number_input("ef-construction-min", min_value=10, max_value=2000, value=50)
                efc_max = col2.number_input("ef-construction-max", min_value=10, max_value=3000, value=500)
                efs_min = col1.number_input("ef-search-min", min_value=2, max_value=2000, value=10)
                efs_max = col2.number_input("ef-search-max", min_value=2, max_value=3000, value=300)
                max_trials = st.number_input("max-trials", min_value=1, max_value=1000, value=50)
                cmd += [
                    "--m-min", str(m_min), "--m-max", str(m_max),
                    "--ef-construction-min", str(efc_min), "--ef-construction-max", str(efc_max),
                    "--ef-search-min", str(efs_min), "--ef-search-max", str(efs_max),
                    "--max-trials", str(max_trials),
                ]
            elif strategy == "bayesian":
                max_trials = st.number_input("max-trials", min_value=1, max_value=5000, value=100)
                startup_trials = st.number_input("startup-trials", min_value=0, max_value=2000, value=10)
                latency_weight = st.number_input("latency-weight", min_value=0.0, value=0.001, step=0.0001, format="%.4f")
                cmd += ["--max-trials", str(max_trials), "--startup-trials", str(startup_trials), "--latency-weight", str(latency_weight)]
            elif strategy == "multi-objective":
                max_trials = st.number_input("max-trials", min_value=1, max_value=5000, value=100)
                min_recall = st.number_input("min-recall", min_value=0.0, max_value=1.0, value=0.99, step=0.01)
                cmd += ["--max-trials", str(max_trials), "--min-recall", str(min_recall)]

        else:
            strategy = st.selectbox("Strategy", ["info", "quick-test", "baseline", "grid-search-cmd", "random-search-cmd"])
            dataset = st.selectbox("Dataset", ["sift1m", "glove100"])
            subset_percent = st.number_input("subset-percent", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
            query_count = st.number_input("query-count", min_value=10, max_value=10000, value=1000)
            output_dir = st.text_input("Output dir", f"results/interactive_dataset/{strategy}")

            cmd = [sys.executable, "-m", "src.dataset_cli", strategy, "--dataset", dataset]

            if strategy != "info":
                cmd += ["--subset-percent", str(subset_percent), "--query-count", str(query_count), "--output-dir", output_dir]

            if strategy == "baseline":
                ef_vals = st.text_input("ef-search-values", "20,40,80,160")
                cmd += ["--ef-search-values", ef_vals]
            elif strategy == "grid-search-cmd":
                grid_points = st.number_input("grid-points", min_value=2, max_value=8, value=3)
                cmd += ["--grid-points", str(grid_points)]
            elif strategy == "random-search-cmd":
                num_trials = st.number_input("num-trials", min_value=1, max_value=5000, value=20)
                cmd += ["--num-trials", str(num_trials)]

        submitted = st.form_submit_button("Run Strategy", type="primary")

    if submitted:
        st.code(" ".join(cmd), language="bash")
        with st.spinner("Running optimization... this may take a while"):
            rc, logs = run_cli_command(cmd)

        if rc == 0:
            st.success("Run completed successfully.")
            st.text_area("CLI output", logs, height=320)
            st.cache_data.clear()
            # Show generated artifacts immediately after a successful run
            if strategy != "info":
                render_run_outputs(REPO_ROOT / output_dir)
        else:
            st.error(f"Run failed with exit code {rc}")
            st.text_area("CLI output", logs, height=320)


def main() -> None:
    apply_professional_theme()
    st.title("📊 HNSW-CS-328 Interactive Report & Hyperparameter Tuning")
    st.caption("Explore reports, inspect results, and launch optimization runs from one dashboard.")

    tab_intro, tab_synth, tab_overview, tab_results, tab_tuning, tab_docs = st.tabs(
        ["Introduction", "Synthetic Visualizer", "Overview", "Results", "HyperTune", "Docs"]
    )

    with tab_intro:
        render_introduction_page()

    with tab_synth:
        render_synthetic_visualizer()

    with tab_overview:
        render_project_summary()
        st.markdown("---")
        st.markdown("### Quick Result Snapshot")
        csv_files = discover_csv_files(RESULTS_ROOT)
        if not csv_files:
            st.info("No result CSV files found under `results/`.")
        else:
            latest = csv_files[0]
            st.write(f"Latest CSV: `{relative(latest)}`")
            df = load_csv(latest)
            st.dataframe(df.head(20), use_container_width=True)

    with tab_results:
        render_results_explorer()

    with tab_tuning:
        render_tuning_runner()

    with tab_docs:
        render_docs_panel()


if __name__ == "__main__":
    main()
