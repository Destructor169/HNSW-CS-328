from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


@dataclass
class DatasetSplit:
    base_vectors: np.ndarray
    query_vectors: np.ndarray


@dataclass
class PreparedDataset:
    split: DatasetSplit
    dataset_label: str = "unknown"


def load_dataset(path: str | Path, delimiter: str = ",") -> np.ndarray:
    """Load vectors from .npy, .npz or .csv files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".npz":
        npz_data = np.load(path)
        if not npz_data.files:
            raise ValueError(f"No arrays found in npz file: {path}")
        data = npz_data[npz_data.files[0]]
    elif suffix == ".csv":
        data = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError("Unsupported file format. Use .npy, .npz, or .csv")

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {data.shape}")
    return data.astype(np.float32, copy=False)


def generate_synthetic_dataset(
    n_vectors: int = 20_000,
    dimension: int = 128,
    n_clusters: int = 40,
    cluster_std: float = 1.5,
    random_state: int = 42,
) -> np.ndarray:
    """Generate synthetic vectors with cluster structure."""
    rng = np.random.default_rng(random_state)

    centers = rng.normal(loc=0.0, scale=8.0, size=(n_clusters, dimension)).astype(np.float32)
    cluster_ids = rng.integers(low=0, high=n_clusters, size=n_vectors)
    noise = rng.normal(loc=0.0, scale=cluster_std, size=(n_vectors, dimension)).astype(np.float32)
    vectors = centers[cluster_ids] + noise
    return vectors.astype(np.float32, copy=False)


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize vectors row-wise."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (vectors / norms).astype(np.float32, copy=False)


def split_dataset(
    vectors: np.ndarray,
    query_fraction: float = 0.1,
    random_state: int = 42,
) -> DatasetSplit:
    """Split vectors into index base vectors and query vectors."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    if not (0.0 < query_fraction < 1.0):
        raise ValueError("query_fraction must be between 0 and 1")

    base, query = train_test_split(
        vectors,
        test_size=query_fraction,
        random_state=random_state,
        shuffle=True,
    )
    return DatasetSplit(
        base_vectors=base.astype(np.float32, copy=False),
        query_vectors=query.astype(np.float32, copy=False),
    )


def compute_ground_truth(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
    save_path: Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact nearest neighbors with brute-force search."""
    if k <= 0:
        raise ValueError("k must be > 0")

    knn = NearestNeighbors(
        n_neighbors=k,
        algorithm="brute",
        metric=metric,
        n_jobs=-1,
    )
    knn.fit(base_vectors)
    distances, indices = knn.kneighbors(query_vectors)

    distances = distances.astype(np.float32, copy=False)
    indices = indices.astype(np.int64, copy=False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, indices)

    return distances, indices


def prepare_dataset_from_config(config: Dict[str, Any], seed: int) -> PreparedDataset:
    """Prepare a dataset split from config for synthetic or file sources."""
    source = str(config.get("source", "synthetic")).strip().lower()
    normalize = bool(config.get("normalize", True))
    query_fraction = float(config.get("query_fraction", 0.1))

    if source == "synthetic":
        synth = config["synthetic"]
        vectors = generate_synthetic_dataset(
            n_vectors=int(synth["n_vectors"]),
            dimension=int(synth["dimension"]),
            n_clusters=int(synth["n_clusters"]),
            cluster_std=float(synth["cluster_std"]),
            random_state=seed,
        )
        if normalize:
            vectors = normalize_vectors(vectors)
        split = split_dataset(vectors=vectors, query_fraction=query_fraction, random_state=seed)
        return PreparedDataset(split=split, dataset_label="synthetic")

    if source == "file":
        file_cfg = config.get("file", {})
        if "path" not in file_cfg:
            raise ValueError("dataset.file.path must be provided when source=file")

        vectors = load_dataset(file_cfg["path"], delimiter=str(file_cfg.get("delimiter", ",")))
        if normalize:
            vectors = normalize_vectors(vectors)
        split = split_dataset(vectors=vectors, query_fraction=query_fraction, random_state=seed)
        return PreparedDataset(split=split, dataset_label=f"file:{file_cfg['path']}")

    raise ValueError("Unsupported dataset source. Use one of: synthetic, file")
