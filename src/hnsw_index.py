from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import hnswlib
except ImportError as import_error:  # pragma: no cover
    hnswlib = None
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


@dataclass
class HNSWBuildResult:
    build_time_seconds: float
    index_size_bytes: int


class HNSWIndexWrapper:
    """Thin wrapper around hnswlib index build and search operations."""

    def __init__(self, space: str = "l2", dimension: int = 128) -> None:
        if hnswlib is None:
            raise ImportError(
                "hnswlib is required for HNSW indexing. Install with `pip install hnswlib`."
            ) from _IMPORT_ERROR

        self.space = space
        self.dimension = dimension
        self.index = hnswlib.Index(space=space, dim=dimension)
        self._is_built = False

    def build_hnsw_index(
        self,
        data: np.ndarray,
        m: int,
        ef_construction: int,
        random_seed: int = 42,
    ) -> HNSWBuildResult:
        if data.ndim != 2 or data.shape[1] != self.dimension:
            raise ValueError(
                f"Expected shape (n, {self.dimension}), got {data.shape}"
            )
        if m <= 0 or ef_construction <= 0:
            raise ValueError("m and ef_construction must be > 0")

        start = time.perf_counter()
        self.index.init_index(
            max_elements=data.shape[0],
            ef_construction=ef_construction,
            M=m,
            random_seed=random_seed,
        )
        self.index.add_items(data, np.arange(data.shape[0], dtype=np.int64))
        build_time = time.perf_counter() - start
        self._is_built = True

        index_size_bytes = int(self.index.index_file_size())
        return HNSWBuildResult(
            build_time_seconds=build_time,
            index_size_bytes=index_size_bytes,
        )

    def search_hnsw(
        self,
        queries: np.ndarray,
        ef_search: int,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if not self._is_built:
            raise RuntimeError("Index must be built before search")
        if k <= 0 or ef_search <= 0:
            raise ValueError("k and ef_search must be > 0")

        self.index.set_ef(ef_search)
        start = time.perf_counter()
        labels, distances = self.index.knn_query(queries, k=k)
        elapsed = time.perf_counter() - start

        labels = labels.astype(np.int64, copy=False)
        distances = distances.astype(np.float32, copy=False)
        return labels, distances, elapsed

    def get_index_stats(self) -> Dict[str, int | bool | str]:
        return {
            "space": self.space,
            "dimension": self.dimension,
            "built": self._is_built,
            "current_count": int(self.index.get_current_count()) if self._is_built else 0,
        }
