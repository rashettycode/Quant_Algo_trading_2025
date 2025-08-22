# Architecture

```
data → features → modeling → predictions → simulations → reports
```

- **src/quant_trader/io**: loaders and storage (Parquet).
- **src/quant_trader/features**: TA, breadth, regimes, feature assembly.
- **src/quant_trader/modeling**: datasets, baselines, advanced, tuning, inference.
- **src/quant_trader/simulation**: vectorized & exact sims, metrics, strategies.
- **scripts/**: thin CLI wrappers for each stage.
