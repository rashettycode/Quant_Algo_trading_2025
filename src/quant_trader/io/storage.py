import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.quant_trader.utils.config import load_config
from src.quant_trader.io.loaders import fetch_all
from src.quant_trader.io.storage import save_parquet

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    df = fetch_all(cfg)

    if df is None or df.empty:
        print("No data downloaded. Check tickers/dates in configs/base.yaml.")
    else:
        save_parquet(df, "data/processed/prices.parquet")
        print(df.head())
        print(f"Saved {len(df):,} rows → data/processed/prices.parquet")

