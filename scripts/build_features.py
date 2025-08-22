# scripts/build_features.py
import sys, argparse, pathlib, pandas as pd
repo = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo))

from pathlib import Path
from src.quant_trader.utils.config import load_config
from src.quant_trader.features.feature_set import build_feature_matrix

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc_dir = Path("data/processed")
    df = pd.read_parquet(proc_dir / "prices.parquet")

    X, y, meta = build_feature_matrix(df, cfg)
    feat = X.copy(); feat["target"] = y; feat = feat.reset_index()
    feat.to_parquet(proc_dir / "features.parquet", index=False)
    print(f"[features] saved {proc_dir/'features.parquet'} rows={len(feat)}")

