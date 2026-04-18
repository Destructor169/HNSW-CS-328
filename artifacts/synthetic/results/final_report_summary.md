# Final Report Summary

## Dataset
- Source: synthetic
- Base vectors: 18000
- Query vectors: 2000
- Dimension: 128

## Best Learned Parameters
- M: 44
- efConstruction: 90
- efSearch: 152

## Performance Comparison

| configuration | recall | latency_ms | memory_bytes | build_time_s |
|---|---:|---:|---:|---:|
| default | 0.9997 | 0.0570 | 11892212 | 2.8698 |
| learned | 1.0000 | 0.0846 | 15925056 | 1.5430 |

## Notes
- Learned parameters are selected by maximizing score = recall - lambda * latency_ms.
- Use results and plots for trade-off analysis and further tuning.