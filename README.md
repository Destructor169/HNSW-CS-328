# HNSW Optimization: Hierarchical Navigable Small World

## 🎯 Project Overview

This project implements and optimizes **HNSW (Hierarchical Navigable Small World)**, an approximate nearest neighbor (ANN) search algorithm. We explore hyperparameter optimization techniques on both synthetic and real-world datasets to understand trade-offs between search accuracy, speed, and memory efficiency.

**Project Goal**: Develop a comprehensive framework for HNSW hyperparameter optimization and demonstrate effectiveness across multiple optimization strategies and datasets.

---

## 👥 Team Members

- **Aditya Kumar**
- **Divisht**
- **Yash Patel**
- **Harshita Singh**

**Status**: ✅ Production Ready | **Version**: 1.0 | **Date**: April 21, 2026

---

## 🔬 What We're Doing

### Phase 1: Synthetic Data Optimization ✅
- Generated 23 synthetic datasets with varying characteristics
- Implemented 5 optimization strategies
- Created 84 configurations with comprehensive metrics
- Generated publication-quality visualizations

### Phase 2: Real Data Optimization ✅
- Built Dataset CLI for SIFT-1M (1M vectors) and GloVe-100 (1.2M vectors)
- Tested subset sampling (2%, 5%, 10%) with proper ground truth handling
- Verified ground truth validity (99.4% queries retain valid neighbors)
- Generated real results with recall@10: 0.48-0.50

---

## ⚙️ Optimizable Parameters

- `M` - Connectivity parameter controlling graph branching (default: 16, optimize: 8-48)
- `efConstruction` - Build-time parameter controlling index quality (default: 200, optimize: 100-400)
- `efSearch` - Query-time parameter controlling recall/latency trade-off (default: 100, optimize: 20-200)

---

## 🔬 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Project Structure                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Data Loading & Preprocessing                │   │
│  │  • Synthetic dataset generation (numpy)            │   │
│  │  • Real dataset loading (SIFT-1M, GloVe-100)       │   │
│  │  • Smart subset sampling with GT filtering         │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │     HNSW Index Building & Searching                │   │
│  │  • hnswlib python bindings                         │   │
│  │  • L2 distance metric                              │   │
│  │  • Index serialization                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │    Hyperparameter Optimization Strategies           │   │
│  │  • Baseline (fixed params, sweep efSearch)        │   │
│  │  • Grid Search (exhaustive 3D grid)               │   │
│  │  • Random Search (random sampling)                │   │
│  │  • Bayesian Optimization (GP-based)               │   │
│  │  • Multi-Objective (Pareto frontier)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │      Metrics & Evaluation                          │   │
│  │  • Recall@1, @10, @100 (accuracy)                 │   │
│  │  • Latency p50/p95/p99 (speed)                    │   │
│  │  • QPS, Build time, Memory (resources)            │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │    Visualization & Analysis                         │   │
│  │  • 5 standard comparison plots per experiment      │   │
│  │  • CSV result export for analysis                  │   │
│  │  • Pareto frontier identification                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Overview

### Real Datasets

#### SIFT-1M (Scale-Invariant Feature Transform)
- **Total Vectors**: 1,000,000 × 128 dimensions
- **Query Set**: 10,000 vectors
- **Ground Truth**: 100 nearest neighbors per query
- **Location**: `data/benchmarks/sift1m/`
- **Use Case**: Image feature similarity search

#### GloVe-100 (Global Vectors)
- **Total Vectors**: 1,193,514 × 100 dimensions
- **Query Set**: 10,000 vectors
- **Ground Truth**: 100 nearest neighbors per query
- **Location**: `data/benchmarks/glove100/`
- **Use Case**: Word embedding similarity search

### Synthetic Datasets
- **23 datasets** with varying sizes (10K-100K vectors)
- **Dimensions**: 50-300D
- **Query sets**: 1K vectors per dataset
- **Generation**: Uniform, Gaussian, Clustered distributions

---

## ✨ Key Features

✅ **Multi-Dataset Support** - Synthetic, SIFT-1M, GloVe-100  
✅ **5 Optimization Strategies** - Baseline, Grid, Random, Bayesian, Multi-Objective  
✅ **Comprehensive Metrics** - Recall@1/10/100, Latency p50/95/99, QPS, Memory  
✅ **Automatic Visualization** - 5 publication-quality plots per experiment  
✅ **Ground Truth Validation** - 99.4% query validity on real subsets  
✅ **Production Ready** - 100% type hints, full error handling  
✅ **Well Documented** - Complete README, Architecture, and Results files  

---

## 🚀 Quick Start: Running the Code

### 1. Installation

```bash
cd /Users/destructor/Desktop/Intro\ To\ DS/Project/HNSW-CS-328
pip install -r requirements.txt
```

### 2. View Dataset Information

```bash
# Check SIFT-1M info
python -m src.dataset_cli info --dataset sift1m

# Check GloVe-100 info
python -m src.dataset_cli info --dataset glove100
```

### 3. Run Quick Test (< 2 minutes)

```bash
python -m src.dataset_cli quick-test --dataset sift1m --subset-percent 2.0
```

### 4. Run Baseline Optimization (< 10 minutes)

```bash
python -m src.dataset_cli baseline \
  --dataset sift1m \
  --subset-percent 5.0 \
  --query-count 2000 \
  --ef-search-values "20,80,160"
```

**Output**: 
- CSV file: `results/dataset_results/baseline/baseline_*.csv`
- Plots: `results/dataset_results/baseline/plots/*.png`

### 5. Run Grid Search (< 30 minutes)

```bash
python -m src.dataset_cli grid-search-cmd \
  --dataset sift1m \
  --subset-percent 5.0 \
  --query-count 2000 \
  --grid-points 3
```

**Grid Configuration**:
- M: [8, 28, 48] (connection count)
- efConstruction: [100, 250, 400] (construction parameter)
- efSearch: [20, 110, 200] (search parameter)
- **Total**: 3 × 3 × 3 = 27 configurations

### 6. Run Random Search (< 20 minutes)

```bash
python -m src.dataset_cli random-search-cmd \
  --dataset sift1m \
  --subset-percent 5.0 \
  --query-count 2000 \
  --num-trials 20
```

---

## 📋 Command Reference

| Command | Purpose | Time |
|---------|---------|------|
| `info` | Display dataset statistics | < 1s |
| `quick-test` | Single configuration test | 2 min |
| `baseline` | Sweep efSearch parameter | 5-10 min |
| `grid-search-cmd` | Exhaustive 3D grid | 15-30 min |
| `random-search-cmd` | Random sampling search | 5-20 min |

**All commands**: `python -m src.dataset_cli [COMMAND] [OPTIONS]`

---

## 📚 Documentation Files

- **README.md** (this file) - Project overview, team, quick start, commands
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed code workflow, implementation details
- **[RESULTS.md](RESULTS.md)** - Real results with embedded plots and analysis

---

## 📈 Expected Results

### Real Data Results (SIFT-1M)
- **Dataset**: 1M vectors × 128D
- **Subset tested**: 5% (50K vectors)
- **Best recall@10**: 0.4967
- **Query latency**: ~50ms p50
- **Ground truth validity**: 99.4%

For complete results and plot analysis, see **[RESULTS.md](RESULTS.md)**

---

## ⚡ Performance Tips

### For Faster Testing
```bash
python -m src.dataset_cli quick-test --subset-percent 0.5
```

### For More Comprehensive Results
```bash
python -m src.dataset_cli grid-search-cmd \
  --subset-percent 10.0 \
  --grid-points 4  # 64 configurations
```

### For Memory Constraints
```bash
python -m src.dataset_cli baseline \
  --subset-percent 2.0 \
  --query-count 500
```

---

## 📂 Project Structure

```
HNSW-CS-328/
├── src/
│   ├── dataset_cli.py          # Main CLI for real data optimization
│   ├── experiments.py          # Synthetic data experiments
│   ├── evaluation.py           # Metrics calculation
│   ├── hnsw_index.py          # Index operations
│   ├── visualization.py        # Plot generation
│   ├── optimization.py         # Optimization algorithms
│   ├── exact_knn.py           # Exact KNN for ground truth
│   └── dataset_loader.py      # Dataset utilities
│
├── data/
│   └── benchmarks/
│       ├── sift1m/            # SIFT-1M dataset
│       └── glove100/          # GloVe-100 dataset
│
├── results/
│   └── dataset_results/       # Real data results with plots
│
├── README.md                  # This file
├── ARCHITECTURE.md            # Code architecture & workflow
└── RESULTS.md                # Real results with plots
```

---

## 📞 Support & Documentation

For detailed implementation questions → **[ARCHITECTURE.md](ARCHITECTURE.md)**

For result analysis and plot explanations → **[RESULTS.md](RESULTS.md)**

---

## 🎓 Learning Path

1. **Read this README** (5 min) - Project overview and quick start
2. **Run a quick test** (2 min) - `python -m src.dataset_cli quick-test`
3. **Read [ARCHITECTURE.md](ARCHITECTURE.md)** (20 min) - Understand how it works
4. **Run experiments** (30+ min) - Explore baseline, grid search, random search
5. **Read [RESULTS.md](RESULTS.md)** (15 min) - Analyze results and plots

---

## 🎯 Learning Resources

- [HNSW Paper](https://arxiv.org/abs/1603.09320) - Original algorithm
- [Optuna Docs](https://optuna.readthedocs.io/) - Optimization framework
- [hnswlib Docs](https://github.com/nmslib/hnswlib) - Python bindings

---

## ✉️ Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/hnsw-cs328/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/hnsw-cs328/discussions)
- **Email**: [your-email@example.com]

---

**Made with ❤️ for efficient approximate nearest neighbor search**
