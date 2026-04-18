# Final Report Summary

## Dataset
- Base vectors: 18000
- Query vectors: 2000
- Dimension: 128

## Best Learned Parameters
- M: 56
- efConstruction: 143
- efSearch: 153

## Performance Comparison

| configuration | recall | latency_ms | memory_bytes | build_time_s |
|---|---:|---:|---:|---:|
| default | 0.9997 | 0.0283 | 11892212 | 1.0685 |
| learned | 1.0000 | 0.0380 | 17649720 | 1.3386 |

## Notes
- Learned parameters are selected by maximizing score = recall - lambda * latency_ms.
- Use results and plots for trade-off analysis and further tuning.