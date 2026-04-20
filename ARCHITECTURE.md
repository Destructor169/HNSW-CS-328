# Architecture & Code Workflow

## 🧩 Architecture Diagram

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                           HNSW-CS-328 Pipeline                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Synthetic Data         Real Data (SIFT-1M / GloVe-100)                  │
│        │                         │                                       │
│        └────────────┬────────────┘                                       │
│                     ▼                                                    │
│            Dataset Loading & Sampling                                    │
│      • load base/query/ground-truth arrays                               │
│      • sample a subset of vectors                                        │
│      • remap ground-truth indices for the subset                         │
│                     ▼                                                    │
│            HNSW Index Construction                                       │
│      • build graph with M and efConstruction                             │
│      • store index metadata and memory usage                             │
│                     ▼                                                    │
│            Query Execution & Evaluation                                  │
│      • search with efSearch                                              │
│      • measure recall, latency, QPS, and build time                      │
│                     ▼                                                    │
│            Strategy Orchestration                                         │
│      • baseline, grid search, random search                              │
│      • Bayesian and multi-objective hooks                                │
│                     ▼                                                    │
│            Visualization & Reporting                                     │
│      • generate 5 standard plots                                          │
│      • export CSVs and markdown summaries                                │
│                     ▼                                                    │
│            Results Directory                                              │
│      • CSV metrics                                                        │
│      • PNG plots                                                          │
│      • markdown summaries                                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## 📐 System Architecture

### High-Level Data Flow

```
Input Data (Synthetic/SIFT-1M/GloVe-100)
    ↓
[Dataset Loading & Preprocessing]
    - Load vectors (base, query)
    - Load ground truth (nearest neighbors)
    - Subset sampling with index mapping
    - Normalize if needed
    ↓
[Ground Truth Validation]
    - Filter GT indices for sampled subset
    - Maintain 99.4%+ query validity
    - Calculate exact KNN via brute-force
    ↓
[Optimization Strategy]
    ├── Baseline: Fixed M & efConstruction, sweep efSearch
    ├── Grid Search: Exhaustive 3D grid exploration
    ├── Random Search: Random parameter sampling
    ├── Bayesian: Optuna TPE sampler
    └── Multi-Objective: Pareto frontier exploration
    ↓
[For Each Configuration]
    - Build HNSW index
    - Run queries against index
    - Measure accuracy (recall@k)
    - Measure speed (latency, QPS)
    - Measure resources (build time, memory)
    ↓
[Evaluation & Ranking]
    - Calculate 12 metrics per configuration
    - Identify best configurations
    - Rank by multiple objectives
    ↓
[Visualization & Export]
    - Generate 5 standard plots
    - Export results to CSV
    - Create markdown reports
    ↓
Output (CSV + PNG + MD files)
```

---

## 🏗️ Module Architecture

### Core Modules

#### 1. **dataset_cli.py** (Main Entry Point)
**Purpose**: Command-line interface for real data optimization  
**Key Components**:

```python
class DatasetConfig:
    """Manages dataset loading and sampling"""
    - load_vectors(path): Load .npy files
    - load_and_sample_dataset(): Smart subset + GT filtering
    - validate_subset(): Check ground truth validity

@app.command
async def info(dataset): 
    """Display dataset statistics"""

@app.command
async def quick_test(dataset, subset_percent):
    """Run single configuration"""

@app.command
async def baseline(dataset, subset_percent, ef_search_values):
    """Sweep efSearch parameter"""

@app.command
async def grid_search_cmd(dataset, subset_percent, grid_points):
    """Exhaustive 3D grid search"""

@app.command
async def random_search_cmd(dataset, subset_percent, num_trials):
    """Random parameter sampling"""
```

**Data Flow**:
1. Parse CLI arguments with Typer
2. Create DatasetConfig instance
3. Load dataset and sample subset
4. Call optimization strategy function
5. Export results to CSV and plots

#### 2. **evaluation.py** (Metrics Calculation)
**Purpose**: Compute performance metrics  
**Key Functions**:

```python
def summarize_metrics(predicted, ground_truth, queries, latencies, build_time, memory):
    """Calculate 12 metrics for configuration"""
    Returns EvaluationResult with:
    - recall_at_1, recall_at_10, recall_at_100
    - latency_p50_ms, latency_p95_ms, latency_p99_ms
    - qps (queries per second)
    - build_time_s
    - memory_mb
    - mrr (mean reciprocal rank)
    - ndcg_at_10
```

**Metric Formulas**:
- **Recall@K**: |intersection(predicted_top_k, gt)| / |gt|
- **Latency**: total_search_time / num_queries
- **QPS**: num_queries / total_search_time
- **MRR**: 1/rank of first relevant result

#### 3. **hnsw_index.py** (Index Operations)
**Purpose**: Build and query HNSW indices  
**Key Functions**:

```python
class HNSWIndexWrapper:
    def build_hnsw_index(vectors, m, ef_construction):
        """Build HNSW index with parameters"""
        - Create index with dimensions
        - Set M and efConstruction
        - Add all vectors
        - Return serialized index
    
    def search_hnsw(query_vectors, index, ef_search):
        """Search using built index"""
        - Set efSearch parameter
        - Query each vector
        - Return predicted neighbors and latencies
        - Calculate build time and memory
```

**Parameters**:
- `M`: Connection count (8-48 typical)
- `efConstruction`: Build quality (100-400 typical)
- `efSearch`: Query accuracy (20-200 typical)

#### 4. **visualization.py** (Plot Generation)
**Purpose**: Create comparison visualizations  
**Key Plot Functions**:

```python
def recall_vs_latency(results_df):
    """Scatter: Accuracy vs Speed trade-off"""
    - X-axis: Latency (ms)
    - Y-axis: Recall@10
    - Color: M parameter
    
def pareto_frontier(results_df):
    """Identify non-dominated configurations"""
    - Pareto optimal points
    - Dominated points (grayed out)
    
def recall_vs_ef_search(results_df):
    """Line: Parameter sensitivity for efSearch"""
    
def recall_vs_m(results_df):
    """Line: Parameter sensitivity for M"""
    
def build_time_vs_ef_construction(results_df):
    """Bar: Build efficiency"""
```

**Common Features**:
- High-DPI (180 DPI)
- Consistent color scheme
- Error bars where applicable
- Grid and legend

#### 5. **optimization.py** (Strategy Implementations)
**Purpose**: Run optimization strategies  
**Strategy Functions**:

```python
def baseline_strategy(dataset, ef_search_values):
    """Fixed M=16, efC=200, sweep efSearch"""
    
def grid_search_strategy(dataset, grid_points):
    """Exhaustive 3D grid"""
    - M: linspace(8, 48, grid_points)
    - efConstruction: linspace(100, 400, grid_points)
    - efSearch: linspace(20, 200, grid_points)
    
def random_search_strategy(dataset, num_trials):
    """Random sampling from parameter space"""
    
def bayesian_strategy(dataset, num_trials):
    """Optuna-based optimization (future)"""
    
def multi_objective_strategy(dataset, num_trials):
    """Pareto frontier exploration (future)"""
```

---

## 🔄 Ground Truth Index Mapping

### The Problem
When we sample vectors from a large dataset:
- Ground truth contains indices to **full dataset** (e.g., 1M for SIFT-1M)
- We build index on **subset** (e.g., 50K for 5% sampling)
- Index uses new indices (0 to 49,999)
- Need to map GT indices accordingly

### The Solution

```python
# Step 1: Create mapping from old indices to new indices
index_map = np.full(n_total, -1, dtype=np.int64)  # Initialize with -1
index_map[sample_indices] = np.arange(n_sample)   # Map sampled indices

# Step 2: Filter ground truth using mapping
filtered_gt = {}
for query_idx in range(n_queries):
    valid_neighbors = []
    for neighbor_idx in original_gt[query_idx]:
        new_idx = index_map[neighbor_idx]
        if new_idx >= 0:  # Check if neighbor is in subset
            valid_neighbors.append(int(new_idx))  # Convert to Python int
    filtered_gt[query_idx] = valid_neighbors

# Step 3: Verify validity
valid_queries = sum(1 for gt_list in filtered_gt.values() if len(gt_list) > 0)
validity_percent = (valid_queries / n_queries) * 100
# Typical result: 99.4% of queries retain valid neighbors
```

**Example**:
```
Original GT for Query 0: [5000, 15000, 3000, ...]  (indices to 1M vectors)
Sample indices: [3000, 5000, 15000, ...]
Index map: 3000→0, 5000→1, 15000→2, ...
New GT: [1, 2, 0, ...]  (indices to 50K subset)
```

---

## 📊 Metrics Calculation Workflow

### Step 1: Run Query
```python
# Build index on subset
index = HNSWIndexWrapper.build_hnsw_index(
    vectors=sampled_vectors,  # Shape: (n_sample, n_dims)
    m=16,
    ef_construction=200
)

# Search with queries
predicted_neighbors, latencies = HNSWIndexWrapper.search_hnsw(
    query_vectors=query_vectors,  # Shape: (n_queries, n_dims)
    index=index,
    ef_search=100
)
```

### Step 2: Calculate Recall
```python
recall_at_10 = 0
for query_idx in range(n_queries):
    predicted_top_10 = predicted_neighbors[query_idx][:10]
    gt_neighbors = filtered_gt[query_idx]
    
    intersection = len(set(predicted_top_10) & set(gt_neighbors))
    recall = intersection / len(gt_neighbors)
    recall_at_10 += recall

recall_at_10 /= n_queries  # Average
```

### Step 3: Calculate Latency Percentiles
```python
latencies_sorted = sorted(latencies)
latency_p50 = latencies_sorted[int(0.50 * len(latencies))]
latency_p95 = latencies_sorted[int(0.95 * len(latencies))]
latency_p99 = latencies_sorted[int(0.99 * len(latencies))]

qps = len(latencies) / sum(latencies)  # Queries per second
```

### Step 4: Record Metrics
```python
result = EvaluationResult(
    m=16,
    ef_construction=200,
    ef_search=100,
    recall_at_1=0.5041,
    recall_at_10=0.5037,
    recall_at_100=0.5142,
    latency_p50_ms=0.050,
    latency_p95_ms=0.060,
    latency_p99_ms=0.070,
    qps=19600,
    build_time_s=2.1,
    memory_mb=31.3
)
```

---

## 🔍 Detailed Implementation: Baseline Strategy

```python
async def baseline(
    dataset: str = "sift1m",
    subset_percent: float = 5.0,
    query_count: int = 2000,
    ef_search_values: str = "20,80,160",
    output_dir: str = "results/dataset_results/baseline"
) -> None:
    """Run baseline optimization: fixed M & efC, sweep efSearch"""
    
    # 1. Load configuration
    config = DatasetConfig(dataset)
    
    # 2. Load and sample data
    base_vecs, query_vecs, gt_matrix, n_total = config.load_and_sample_dataset(
        subset_percent=subset_percent,
        query_count=query_count
    )
    
    # 3. Build index once (fixed parameters)
    m = 16
    ef_construction = 200
    index = hnsw.build_hnsw_index(base_vecs, m, ef_construction)
    build_time = measure_time()
    index_memory = index.get_size()
    
    # 4. For each efSearch value
    results = []
    for ef_search in [20, 80, 160]:
        # Query
        predicted, latencies = hnsw.search_hnsw(
            query_vecs, index, ef_search
        )
        
        # Evaluate
        metrics = evaluation.summarize_metrics(
            predicted, gt_matrix, query_vecs,
            latencies, build_time, index_memory
        )
        results.append(metrics)
    
    # 5. Export results
    df = pd.DataFrame([r.dict() for r in results])
    df.to_csv(f"{output_dir}/baseline_*.csv")
    
    # 6. Generate plots
    visualization.generate_plots(df, output_dir)
```

---

## 📈 Data Structures

### EvaluationResult (Dataclass)
```python
@dataclass
class EvaluationResult:
    m: int                          # Connection parameter
    ef_construction: int            # Build parameter
    ef_search: int                  # Search parameter
    
    # Accuracy metrics
    recall_at_1: float              # Top-1 recall
    recall_at_10: float             # Top-10 recall
    recall_at_100: float            # Top-100 recall
    mrr: float                      # Mean reciprocal rank
    
    # Latency metrics (milliseconds)
    latency_p50_ms: float           # 50th percentile
    latency_p95_ms: float           # 95th percentile
    latency_p99_ms: float           # 99th percentile
    
    # Resource metrics
    qps: float                      # Queries per second
    build_time_s: float             # Index build time (seconds)
    memory_mb: float                # Index memory (MB)
```

### ResultsDataFrame (CSV Export)
```
m,ef_construction,ef_search,recall_at_1,recall_at_10,recall_at_100,mrr,latency_p50_ms,latency_p95_ms,latency_p99_ms,qps,build_time_s,memory_mb
16,200,20,0.5041,0.5037,0.5142,0.6182,0.050,0.060,0.070,19600,2.1,31.3
16,200,80,0.5041,0.5037,0.5142,0.6182,0.052,0.062,0.072,19200,2.1,31.3
16,200,160,0.5046,0.5042,0.5147,0.6187,0.055,0.065,0.075,18200,2.1,31.3
```

---

## 🚀 Running a Complete Experiment

### Command Execution Flow

```bash
python -m src.dataset_cli baseline \
  --dataset sift1m \
  --subset-percent 5.0 \
  --query-count 2000 \
  --ef-search-values "20,80,160"
```

### Internal Execution

```
1. Parse Arguments (Typer)
   dataset=sift1m, subset_percent=5.0, query_count=2000
   ef_search_values=[20, 80, 160]

2. Create DatasetConfig
   config = DatasetConfig("sift1m")

3. Load Dataset
   base: (1000000, 128)
   queries: (10000, 128)
   gt: (10000, 100)

4. Sample Subset
   sampled_indices: 50000 random indices
   base: (50000, 128)
   queries: 2000 samples
   gt: filtered to new indices (99.4% valid)

5. Build Index
   M=16, efConstruction=200
   Index built: 2.1 seconds
   Memory: 31.3 MB

6. For each efSearch in [20, 80, 160]:
   a) Search queries
      latencies: per-query times
   b) Calculate metrics
      recall@1, @10, @100
      latency p50, p95, p99
      QPS
   c) Record result

7. Export Results
   CSV: baseline_20260421T100939Z.csv
   3 rows (one per efSearch)

8. Generate Plots
   - recall_vs_latency.png
   - pareto_frontier.png
   - recall_vs_ef_search.png
   - recall_vs_m.png
   - build_time_vs_ef_construction.png
```

---

## 🧪 Testing & Validation

### Ground Truth Validation
```python
def validate_ground_truth(filtered_gt, original_gt, sample_indices):
    """Verify GT filtering correctness"""
    valid_count = 0
    for query_idx, neighbors in filtered_gt.items():
        if len(neighbors) > 0:
            valid_count += 1
    
    validity = valid_count / len(filtered_gt)
    print(f"Ground truth validity: {validity:.1%}")
    # Expected: 99.4%+
```

### Recall Validation
```python
def validate_recall(predicted, ground_truth):
    """Verify recall calculation"""
    for query_idx in range(len(predicted)):
        pred_top_10 = set(predicted[query_idx][:10])
        gt = set(ground_truth[query_idx])
        
        intersection = len(pred_top_10 & gt)
        recall = intersection / len(gt)
        
        assert 0 <= recall <= 1, f"Invalid recall: {recall}"
```

---

## 📊 Result Analysis Workflow

### CSV Analysis
```python
import pandas as pd

df = pd.read_csv("results/dataset_results/baseline/baseline_*.csv")

# Best recall
best_config = df.loc[df['recall_at_10'].idxmax()]
print(f"Best recall@10: {best_config['recall_at_10']:.4f}")
print(f"  M={best_config['m']}, efC={best_config['ef_construction']}, efS={best_config['ef_search']}")

# Speed analysis
fastest = df.loc[df['latency_p50_ms'].idxmin()]
print(f"Fastest: {fastest['latency_p50_ms']:.3f}ms latency")

# Pareto frontier
from scipy.spatial.distance import cdist
pareto_idx = pareto_frontier(df[['recall_at_10', 'latency_p50_ms']].values)
pareto_configs = df.iloc[pareto_idx]
```

---

## 🔗 Integration Points

### With Synthetic Data Pipeline
- Same evaluation.py for metrics
- Same visualization.py for plots
- Different data loading (synthetic vs real)

### With Future Bayesian/Multi-Objective
- Plug into optimization loop
- Use same evaluation.py
- Generate same CSV/PNG outputs

---

## 💾 File I/O Operations

### Reading
```python
# Dataset loading
base_vecs = np.load("data/benchmarks/sift1m/base.npy")
query_vecs = np.load("data/benchmarks/sift1m/query.npy")
gt_matrix = np.load("data/benchmarks/sift1m/gt.npy")

# Results loading
results_df = pd.read_csv("results/dataset_results/baseline/baseline_*.csv")
```

### Writing
```python
# Results export
results_df.to_csv(f"{output_dir}/baseline_{timestamp}.csv", index=False)

# Plot saving
plt.savefig(f"{output_dir}/plots/recall_vs_latency.png", dpi=180)
```

---

## 🎯 Performance Characteristics

### Time Complexity
| Operation | Time |
|-----------|------|
| Load 1M vectors (128D) | ~1-2 sec |
| Build HNSW index | ~2-5 sec |
| Query 2000 vectors | ~50-200 ms |
| Calculate metrics | ~10 ms |
| Generate 5 plots | ~2-5 sec |

### Space Complexity
- Base vectors: 1M × 128 × 8 = 1 GB
- Index structure: ~3% of base vectors (~30 MB)
- Query vectors: 10K × 128 × 8 = 10 MB

---

## 🔐 Error Handling

```python
try:
    config = DatasetConfig(dataset)
    base_vecs, query_vecs, gt, n_total = config.load_and_sample_dataset(...)
except FileNotFoundError:
    raise RuntimeError(f"Dataset {dataset} not found at {config.base_path}")

try:
    index = hnsw.build_hnsw_index(base_vecs, m, ef_construction)
except Exception as e:
    logger.error(f"Failed to build HNSW index: {e}")
    raise

try:
    visualization.generate_plots(df, output_dir)
except Exception as e:
    logger.warning(f"Failed to generate plots: {e}")
    # Continue anyway, results are still valid
```

---

**For real results and plot analysis, see [RESULTS.md](RESULTS.md)**
