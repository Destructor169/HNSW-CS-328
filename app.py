from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml


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
        else:
            st.error(f"Run failed with exit code {rc}")
            st.text_area("CLI output", logs, height=320)


def main() -> None:
    st.title("📊 HNSW-CS-328 Interactive Report & Hyperparameter Tuning")
    st.caption("Explore reports, inspect results, and launch optimization runs from one dashboard.")

    tab_overview, tab_results, tab_tuning, tab_docs = st.tabs(
        ["Overview", "Results", "HyperTune", "Docs"]
    )

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
