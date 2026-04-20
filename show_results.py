import pandas as pd
import os

print("\n" + "="*80)
print("OPTIMIZATION STRATEGIES - DETAILED RESULTS")
print("="*80)

results = {
    'baseline': 'results/final_results/baseline/baseline_20260420T192058Z.csv',
    'grid': 'results/final_results/grid/grid_search_results.csv',
    'random': 'results/final_results/random/random_search_results.csv',
    'bayesian': 'results/final_results/bayesian/bayesian_results.csv',
    'multi_objective': 'results/final_results/multi_objective/multi_objective_pareto.csv'
}

for strategy, filepath in results.items():
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"\n{'='*80}")
        print(f"📊 {strategy.upper()} - {len(df)} configurations")
        print(f"{'='*80}")
        print(f"\nRecall Statistics:")
        print(f"  Best:  {df['recall'].max():.6f}")
        print(f"  Mean:  {df['recall'].mean():.6f}")
        print(f"  Worst: {df['recall'].min():.6f}")
        
        print(f"\nLatency Statistics (seconds):")
        print(f"  Best (fastest):  {df['latency'].min():.4f}s")
        print(f"  Mean:            {df['latency'].mean():.4f}s")
        print(f"  Worst (slowest): {df['latency'].max():.4f}s")
        
        print(f"\nBuild Time Statistics (seconds):")
        print(f"  Best:  {df['build_time'].min():.4f}s")
        print(f"  Mean:  {df['build_time'].mean():.4f}s")
        print(f"  Worst: {df['build_time'].max():.4f}s")
        
        # Best configuration
        best_idx = df['recall'].idxmax()
        best_config = df.loc[best_idx]
        print(f"\n✅ Best Configuration (Max Recall):")
        print(f"   M: {int(best_config['m']):2d}  efConstruction: {int(best_config['ef_construction']):3d}  efSearch: {int(best_config['ef_search']):3d}")
        print(f"   Recall: {best_config['recall']:.6f}  Latency: {best_config['latency']:.4f}s  Build: {best_config['build_time']:.4f}s")
