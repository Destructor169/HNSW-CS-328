# HNSW-CS-328

Learning graph construction for Approximate Nearest Neighbour (ANN) search with HNSW using data-driven parameter optimization.

## Project Goal

Automatically learn effective HNSW parameters:

- M
- efConstruction
- efSearch

to improve Recall@k while controlling latency, build time, and memory footprint.

## Implemented Pipeline

1. Dataset preparation (synthetic or file-based)
2. Exact KNN ground-truth generation using brute-force nearest neighbors
3. HNSW index construction and querying via hnswlib
4. Metric evaluation: recall, latency, build time, memory
5. Bayesian optimization via Optuna (TPE sampler)
6. Multi-objective Optuna optimization (recall, latency, build time, memory)
7. Automatic ranked Pareto report generation
8. Parameter sweep experiments
9. Visualization and Pareto frontier plotting
10. Default vs learned configuration comparison
11. Final report summary generation

## Repository Structure

project/
├── data/
│   ├── raw/
│   ├── embeddings/
│   └── ground_truth/
├── src/
│   ├── dataset_loader.py
│   ├── exact_knn.py
│   ├── hnsw_index.py
│   ├── evaluation.py
│   ├── optimization.py
│   ├── experiments.py
│   └── visualization.py
├── configs/
│   └── default.yaml
├── notebooks/
├── results/
├── plots/
├── main.py
└── requirements.txt

## Setup

### Option A: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: conda

```bash
conda create -n hnsw-opt python=3.10 -y
conda activate hnsw-opt
pip install -r requirements.txt
```

## Run

```bash
python main.py --config configs/default.yaml
```

## Configuration

Edit `configs/default.yaml` to change:

- dataset source (`synthetic` or `file`)
- file data paths
- synthetic dataset shape and dimension
- normalization and split ratio
- default HNSW baseline parameters
- Optuna search ranges and number of trials
- multi-objective trial count and Pareto ranking weights
- minimum recall threshold used for Pareto ranking
- output artifact paths

### Dataset Modes

The pipeline supports two config-driven dataset modes under `dataset.source`:

1. `synthetic`
- Uses `dataset.synthetic.*`
- Generates vectors and splits by `query_fraction`

2. `file`
- Uses `dataset.file.path`
- Loads a single matrix (`.npy`, `.npz`, `.csv`) and splits by `query_fraction`

## Outputs

After running the pipeline, artifacts are written to:

- CSV results in `results/`
- figures in `plots/`
- report in `results/final_report_summary.md`

Key files:

- `results/optimization_history.csv`
- `results/multi_objective_history.csv`
- `results/pareto_ranked_report.csv`
- `results/pareto_ranked_report.md`
- `results/parameter_sweeps.csv`
- `results/default_vs_learned.csv`
- `plots/recall_vs_latency.png`
- `plots/pareto_frontier.png`

## Notes

- Ground truth is computed using exact brute-force nearest neighbors.
- Optimization score is:

	score = recall - lambda * latency_ms

	where lambda is configurable in `configs/default.yaml`.
