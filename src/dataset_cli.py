"""
Dataset CLI for HNSW optimization with real datasets (SIFT-1M and GloVe-100).
Supports sampling subsets and running all optimization strategies.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import typer

from src.evaluation import summarize_metrics
from src.hnsw_index import HNSWIndexWrapper
from src.visualization import (
    plot_recall_vs_latency,
    plot_pareto_frontier,
    plot_recall_vs_ef_search,
    plot_recall_vs_m,
    plot_build_time_vs_ef_construction,
)


app = typer.Typer(help="📊 Dataset-based HNSW Optimization with Real Data")


class DatasetConfig:
    """Configuration for dataset loading and sampling."""
    
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Path,
        subset_percent: float = 2.0,
        query_count: int = 1000,
        random_state: int = 42,
    ):
        self.dataset_name = dataset_name
        self.dataset_path = Path(dataset_path)
        self.subset_percent = subset_percent
        self.query_count = query_count
        self.random_state = random_state
        
        # Validate dataset path
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
    
    def load_vectors(self, file_type: str = "base") -> np.ndarray:
        """Load vectors from numpy file."""
        filepath = self.dataset_path / f"{file_type}.npy"
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        vectors = np.load(filepath)
        print(f"  Loaded {file_type}: shape={vectors.shape}, dtype={vectors.dtype}")
        return vectors.astype(np.float32)
    
    def load_and_sample_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load base, query, and ground truth vectors with sampling.
        
        Returns:
            (base_vectors_sampled, query_vectors, ground_truth_filtered)
        """
        print(f"\n{'='*70}")
        print(f"📂 Loading Dataset: {self.dataset_name}")
        print(f"{'='*70}")
        
        # Load base vectors
        print(f"\n1️⃣  Loading base vectors...")
        base_vectors = self.load_vectors("base")
        
        # Load query vectors
        print(f"\n2️⃣  Loading query vectors...")
        query_vectors = self.load_vectors("query")
        
        # Load ground truth (indices to base vectors)
        print(f"\n3️⃣  Loading ground truth...")
        ground_truth_full = self.load_vectors("ground_truth")
        
        # Sample base vectors
        n_total = len(base_vectors)
        n_sample = max(1000, int(n_total * self.subset_percent / 100))
        
        print(f"\n4️⃣  Sampling base vectors...")
        print(f"  Original: {n_total:,} vectors")
        print(f"  Sampling: {self.subset_percent}% = {n_sample:,} vectors")
        
        rng = np.random.default_rng(self.random_state)
        sample_indices = np.sort(rng.choice(n_total, size=n_sample, replace=False))
        base_vectors_sampled = base_vectors[sample_indices]
        
        # Create mapping from old indices to new indices
        # index_map[old_idx] = new_idx (or -1 if not in sample)
        index_map = np.full(n_total, -1, dtype=np.int64)
        index_map[sample_indices] = np.arange(n_sample)
        
        # Filter ground truth: keep only neighbors that are in the sampled set
        print(f"\n4️⃣ Filtering ground truth indices...")
        ground_truth_filtered = np.full_like(ground_truth_full, -1, dtype=np.int64)
        valid_count = 0
        
        for q_idx, neighbors in enumerate(ground_truth_full):
            filtered_neighbors = []
            for neighbor_idx_val in neighbors:
                neighbor_idx = int(neighbor_idx_val)  # Ensure it's a Python int
                if neighbor_idx < len(index_map):
                    new_idx = index_map[neighbor_idx]
                    if new_idx >= 0:  # Neighbor is in sampled set
                        filtered_neighbors.append(new_idx)
            
            # Fill with valid neighbors (pad with -1 if fewer than original)
            if filtered_neighbors:
                ground_truth_filtered[q_idx, :len(filtered_neighbors)] = filtered_neighbors
                valid_count += 1
        
        print(f"  Queries with valid neighbors: {valid_count}/{len(ground_truth_full)}")
        
        # Filter query vectors if specified
        if self.query_count and len(query_vectors) > self.query_count:
            print(f"\n5️⃣  Sampling query vectors: {len(query_vectors)} → {self.query_count}")
            query_indices = rng.choice(len(query_vectors), size=self.query_count, replace=False)
            query_vectors = query_vectors[query_indices]
            ground_truth_filtered = ground_truth_filtered[query_indices]
        
        print(f"\n✅ Final shapes:")
        print(f"   Base:   {base_vectors_sampled.shape}")
        print(f"   Query:  {query_vectors.shape}")
        print(f"   GT:     {ground_truth_filtered.shape}")
        
        return base_vectors_sampled, query_vectors, ground_truth_filtered


def _ensure_output_dir(path: Path) -> Path:
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_timestamp_str() -> str:
    """Get ISO 8601 timestamp string."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _run_single_configuration(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth: np.ndarray,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> Optional[dict]:
    """
    Run HNSW with a single configuration and return metrics.
    
    Returns:
        Dictionary with performance metrics or None on failure
    """
    try:
        # Get dimensions from data
        dimension = base_vectors.shape[1]
        
        # Build index
        start_build = time.time()
        index_wrapper = HNSWIndexWrapper(space="l2", dimension=dimension)
        build_result = index_wrapper.build_hnsw_index(
            data=base_vectors,
            m=m,
            ef_construction=ef_construction,
        )
        build_time = build_result.build_time_seconds
        memory_bytes = build_result.index_size_bytes
        
        # Search and evaluate
        start_search = time.time()
        predicted_indices, predicted_distances, search_time = index_wrapper.search_hnsw(
            queries=query_vectors,
            ef_search=ef_search,
            k=100,
        )
        
        # Convert search time to milliseconds per query
        query_latencies_ms = [(search_time * 1000) / len(query_vectors)] * len(query_vectors)
        
        # Summarize metrics
        results = summarize_metrics(
            predicted=predicted_indices,
            ground_truth=ground_truth.astype(np.int64),
            query_time_seconds=search_time,
            query_latencies_ms=query_latencies_ms,
            num_queries=len(query_vectors),
            build_time_seconds=build_time,
            memory_bytes=memory_bytes,
            k=100,
        )
        
        return {
            "m": m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "recall_at_1": results.recall_at_1,
            "recall_at_10": results.recall_at_10,
            "recall_at_100": results.recall_at_100,
            "mrr_at_k": results.mrr_at_k,
            "latency_p50_ms": results.latency_p50_ms,
            "latency_p95_ms": results.latency_p95_ms,
            "latency_p99_ms": results.latency_p99_ms,
            "qps": results.qps,
            "build_time_s": build_time,
            "memory_mb": results.memory_mb,
        }
    except Exception as e:
        print(f"  ❌ Configuration failed: {e}")
        return None


@app.command()
def baseline(
    dataset: str = typer.Option(
        "sift1m",
        help="Dataset name: 'sift1m' or 'glove100'",
    ),
    subset_percent: float = typer.Option(
        2.0,
        help="Percentage of dataset to use (e.g., 2.0 for 2%)",
    ),
    query_count: int = typer.Option(
        1000,
        help="Number of query vectors to use",
    ),
    ef_search_values: str = typer.Option(
        "20,40,80,160",
        help="Comma-separated efSearch values to sweep",
    ),
    output_dir: str = typer.Option(
        "results/dataset_results/baseline",
        help="Output directory for results",
    ),
) -> None:
    """Run baseline strategy on real dataset."""
    
    # Setup dataset
    dataset_path = Path("data/benchmarks") / dataset
    config = DatasetConfig(
        dataset_name=dataset,
        dataset_path=dataset_path,
        subset_percent=subset_percent,
        query_count=query_count,
    )
    
    base_vectors, query_vectors, ground_truth = config.load_and_sample_dataset()
    output_path = _ensure_output_dir(Path(output_dir))
    
    print(f"\n{'='*70}")
    print(f"🔍 BASELINE Strategy on Real Data")
    print(f"{'='*70}\n")
    
    # Fixed parameters
    m = 16
    ef_construction = 200
    ef_search_list = [int(x.strip()) for x in ef_search_values.split(",")]
    
    results = []
    for i, ef_search in enumerate(ef_search_list, 1):
        print(f"\n[{i}/{len(ef_search_list)}] Testing efSearch={ef_search}...")
        result = _run_single_configuration(
            base_vectors, query_vectors, ground_truth,
            m, ef_construction, ef_search
        )
        if result:
            results.append(result)
            print(f"  ✅ Recall@10: {result['recall_at_10']:.4f}")
    
    if not results:
        print("❌ No results generated!")
        return
    
    # Save results
    df = pd.DataFrame(results)
    output_file = output_path / f"baseline_{_get_timestamp_str()}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Generate plots
    _generate_plots(df, output_path, dataset)


@app.command()
def grid_search_cmd(
    dataset: str = typer.Option(
        "sift1m",
        help="Dataset name: 'sift1m' or 'glove100'",
    ),
    subset_percent: float = typer.Option(
        2.0,
        help="Percentage of dataset to use",
    ),
    query_count: int = typer.Option(
        1000,
        help="Number of query vectors to use",
    ),
    grid_points: int = typer.Option(
        3,
        help="Number of points per parameter dimension",
    ),
    output_dir: str = typer.Option(
        "results/dataset_results/grid",
        help="Output directory for results",
    ),
) -> None:
    """Run grid search on real dataset."""
    
    # Setup dataset
    dataset_path = Path("data/benchmarks") / dataset
    config = DatasetConfig(
        dataset_name=dataset,
        dataset_path=dataset_path,
        subset_percent=subset_percent,
        query_count=query_count,
    )
    
    base_vectors, query_vectors, ground_truth = config.load_and_sample_dataset()
    output_path = _ensure_output_dir(Path(output_dir))
    
    print(f"\n{'='*70}")
    print(f"🔍 GRID SEARCH on Real Data")
    print(f"{'='*70}\n")
    
    # Define parameter grid
    m_values = np.linspace(8, 48, grid_points, dtype=int)
    ef_construction_values = np.linspace(100, 400, grid_points, dtype=int)
    ef_search_values = np.linspace(20, 200, grid_points, dtype=int)
    
    results = []
    total = len(m_values) * len(ef_construction_values) * len(ef_search_values)
    count = 0
    
    for m in m_values:
        for ef_c in ef_construction_values:
            for ef_s in ef_search_values:
                count += 1
                print(f"\n[{count}/{total}] M={m}, efC={ef_c}, efS={ef_s}...")
                result = _run_single_configuration(
                    base_vectors, query_vectors, ground_truth,
                    int(m), int(ef_c), int(ef_s)
                )
                if result:
                    results.append(result)
                    print(f"  ✅ Recall@10: {result['recall_at_10']:.4f}")
    
    if not results:
        print("❌ No results generated!")
        return
    
    # Save results
    df = pd.DataFrame(results)
    output_file = output_path / f"grid_search_{_get_timestamp_str()}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Generate plots
    _generate_plots(df, output_path, dataset)


@app.command()
def random_search_cmd(
    dataset: str = typer.Option(
        "sift1m",
        help="Dataset name: 'sift1m' or 'glove100'",
    ),
    subset_percent: float = typer.Option(
        2.0,
        help="Percentage of dataset to use",
    ),
    query_count: int = typer.Option(
        1000,
        help="Number of query vectors to use",
    ),
    num_trials: int = typer.Option(
        20,
        help="Number of random configurations to try",
    ),
    output_dir: str = typer.Option(
        "results/dataset_results/random",
        help="Output directory for results",
    ),
) -> None:
    """Run random search on real dataset."""
    
    # Setup dataset
    dataset_path = Path("data/benchmarks") / dataset
    config = DatasetConfig(
        dataset_name=dataset,
        dataset_path=dataset_path,
        subset_percent=subset_percent,
        query_count=query_count,
    )
    
    base_vectors, query_vectors, ground_truth = config.load_and_sample_dataset()
    output_path = _ensure_output_dir(Path(output_dir))
    
    print(f"\n{'='*70}")
    print(f"🔍 RANDOM SEARCH on Real Data")
    print(f"{'='*70}\n")
    
    rng = np.random.default_rng(42)
    results = []
    
    for trial in range(1, num_trials + 1):
        # Random configuration
        m = int(rng.integers(8, 48))
        ef_c = int(rng.integers(100, 401))
        ef_s = int(rng.integers(20, 201))
        
        print(f"\n[{trial}/{num_trials}] M={m}, efC={ef_c}, efS={ef_s}...")
        result = _run_single_configuration(
            base_vectors, query_vectors, ground_truth,
            m, ef_c, ef_s
        )
        if result:
            results.append(result)
            print(f"  ✅ Recall@10: {result['recall_at_10']:.4f}")
    
    if not results:
        print("❌ No results generated!")
        return
    
    # Save results
    df = pd.DataFrame(results)
    output_file = output_path / f"random_search_{_get_timestamp_str()}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Generate plots
    _generate_plots(df, output_path, dataset)


@app.command()
def quick_test(
    dataset: str = typer.Option(
        "sift1m",
        help="Dataset name: 'sift1m' or 'glove100'",
    ),
    subset_percent: float = typer.Option(
        2.0,
        help="Percentage of dataset to use",
    ),
    output_dir: str = typer.Option(
        "results/dataset_results/quick_test",
        help="Output directory for results",
    ),
) -> None:
    """Quick test with a single configuration."""
    
    # Setup dataset
    dataset_path = Path("data/benchmarks") / dataset
    config = DatasetConfig(
        dataset_name=dataset,
        dataset_path=dataset_path,
        subset_percent=subset_percent,
        query_count=500,
    )
    
    base_vectors, query_vectors, ground_truth = config.load_and_sample_dataset()
    output_path = _ensure_output_dir(Path(output_dir))
    
    print(f"\n{'='*70}")
    print(f"⚡ Quick Test - Single Configuration")
    print(f"{'='*70}\n")
    
    # Single configuration
    m, ef_c, ef_s = 16, 200, 100
    
    print(f"\nRunning configuration: M={m}, efConstruction={ef_c}, efSearch={ef_s}...")
    result = _run_single_configuration(
        base_vectors, query_vectors, ground_truth,
        m, ef_c, ef_s
    )
    
    if result:
        print(f"\n✅ Test completed successfully!")
        print(f"\nResults:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:12.4f}")
            else:
                print(f"  {key:20s}: {value}")
        
        # Save result
        df = pd.DataFrame([result])
        output_file = output_path / f"quick_test_{_get_timestamp_str()}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Result saved to: {output_file}")
        
        # Generate plots
        _generate_plots(df, output_path, dataset)
    else:
        print("❌ Test failed!")


def _generate_plots(
    results_df: pd.DataFrame,
    output_path: Path,
    dataset: str,
) -> None:
    """Generate all comparison plots."""
    print(f"\n{'='*70}")
    print(f"📊 Generating Plots")
    print(f"{'='*70}\n")
    
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Recall vs Latency
        print("  📈 recall_vs_latency.png")
        plot_recall_vs_latency(
            results_df,
            output_path=plots_dir / "recall_vs_latency.png",
        )
        
        # 2. Pareto Frontier
        print("  📈 pareto_frontier.png")
        plot_pareto_frontier(
            results_df,
            output_path=plots_dir / "pareto_frontier.png",
        )
        
        # 3. Recall vs efSearch (if column exists)
        if "ef_search" in results_df.columns:
            print("  📈 recall_vs_ef_search.png")
            plot_recall_vs_ef_search(
                results_df,
                output_path=plots_dir / "recall_vs_ef_search.png",
            )
        
        # 4. Recall vs M (if column exists)
        if "m" in results_df.columns:
            print("  📈 recall_vs_m.png")
            plot_recall_vs_m(
                results_df,
                output_path=plots_dir / "recall_vs_m.png",
            )
        
        # 5. Build Time vs efConstruction (if column exists)
        if "ef_construction" in results_df.columns:
            print("  📈 build_time_vs_ef_construction.png")
            plot_build_time_vs_ef_construction(
                results_df,
                output_path=plots_dir / "build_time_vs_ef_construction.png",
            )
        
        print(f"\n✅ All plots saved to: {plots_dir}")
        
    except Exception as e:
        print(f"  ⚠️  Error generating plots: {e}")


@app.command()
def info(
    dataset: str = typer.Option(
        "sift1m",
        help="Dataset name: 'sift1m' or 'glove100'",
    ),
) -> None:
    """Display dataset information."""
    dataset_path = Path("data/benchmarks") / dataset
    
    print(f"\n{'='*70}")
    print(f"📊 Dataset Information: {dataset.upper()}")
    print(f"{'='*70}\n")
    
    config = DatasetConfig(
        dataset_name=dataset,
        dataset_path=dataset_path,
        subset_percent=100.0,  # Don't sample for info
        query_count=0,  # Don't filter for info
    )
    
    try:
        base = config.load_vectors("base")
        query = config.load_vectors("query")
        gt = config.load_vectors("ground_truth")
        
        print(f"\n📈 Statistics:")
        print(f"  Base vectors:  {base.shape[0]:>12,} × {base.shape[1]:<4} dims")
        print(f"  Query vectors: {query.shape[0]:>12,} × {query.shape[1]:<4} dims")
        print(f"  Ground truth:  {gt.shape[0]:>12,} × {gt.shape[1]:<4} columns")
        
        print(f"\n📊 Subset Examples:")
        for percent in [0.5, 1.0, 2.0, 5.0]:
            n_sample = max(1000, int(base.shape[0] * percent / 100))
            print(f"  {percent}% sampling: {n_sample:>8,} vectors")
        
        print(f"\n✅ Dataset ready for optimization!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    app()
