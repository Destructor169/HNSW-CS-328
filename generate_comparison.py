#!/usr/bin/env python3
"""Generate comprehensive comparison of all optimization strategies."""

import pandas as pd
import json
from pathlib import Path

def main():
    strategies = {
        'baseline': 'results/final_results/baseline/baseline_results.csv',
        'grid': 'results/final_results/grid/grid_results.csv',
        'random': 'results/final_results/random/random_results.csv',
        'bayesian': 'results/final_results/bayesian/bayesian_results.csv',
        'multi_objective': 'results/final_results/multi_objective/multi_objective_pareto.csv'
    }
    
    summary = {}
    print("=" * 80)
    print("OPTIMIZATION STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    
    for strategy, filepath in strategies.items():
        df = pd.read_csv(filepath)
        
        best_recall = df['recall'].max()
        best_latency = df['latency'].min()
        best_build_time = df['build_time'].min()
        num_trials = len(df)
        
        summary[strategy] = {
            'num_trials': num_trials,
            'best_recall': float(best_recall),
            'best_latency': float(best_latency),
            'best_build_time': float(best_build_time),
            'avg_recall': float(df['recall'].mean()),
            'avg_latency': float(df['latency'].mean()),
        }
        
        print(f"\n📊 {strategy.upper()}")
        print(f"   Trials: {num_trials}")
        print(f"   Best Recall: {best_recall:.6f}")
        print(f"   Best Latency: {best_latency:.4f}s")
        print(f"   Best Build Time: {best_build_time:.4f}s")
        print(f"   Avg Recall: {df['recall'].mean():.6f}")
        print(f"   Avg Latency: {df['latency'].mean():.4f}s")
    
    # Save summary to JSON
    with open('results/final_results/comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ Summary saved to comparison_summary.json")

if __name__ == '__main__':
    main()
