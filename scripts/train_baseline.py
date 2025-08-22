# scripts/train_baseline.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.quant_trader.modeling.baselines import run_baseline

if __name__ == "__main__":
    m = run_baseline(
        features_path="data/processed/features.parquet",
        out_path="outputs/predictions/baseline.parquet",
        max_depth=3,
        test_quantile=0.8,
        random_state=42,
    )
    print("Baseline metrics:", m)
    print("Saved predictions → outputs/predictions/baseline.parquet")
