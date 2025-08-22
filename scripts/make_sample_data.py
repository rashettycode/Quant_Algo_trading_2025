# scripts/make_sample_data.py
import pathlib
import pandas as pd
import numpy as np
import shutil
import argparse

root = pathlib.Path(__file__).resolve().parents[1]
sample_dir = root / "data" / "sample"
processed_dir = root / "data" / "processed"
sample_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-processed", action="store_true",
                        help="Also copy sample files to data/processed/")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite processed files if they exist")
    args = parser.parse_args()

    np.random.seed(42)

    # Prices sample (2 tickers, ~20 rows)
    dates = pd.date_range("2024-12-01", periods=20, freq="B")
    rows = []
    for t in ["AAPL", "MSFT"]:
        price = 100.0 + np.cumsum(np.random.randn(len(dates)))
        for d, p in zip(dates, price):
            rows.append({
                "ticker": t,
                "date": d,
                "open": p * 0.995,
                "high": p * 1.005,
                "low":  p * 0.990,
                "close": p,
                "adj_close": p,
                "volume": int(1e6 + np.random.randint(0, 5e5)),
            })
    prices = pd.DataFrame(rows).sort_values(["ticker","date"])
    prices_path = sample_dir / "prices_sample.parquet"
    prices.to_parquet(prices_path, index=False)

    # Features sample
    feat = []
    for t in ["AAPL", "MSFT"]:
        df_t = prices[prices["ticker"] == t].copy()
        df_t["ret_1d"] = np.log(df_t["close"]).diff()
        df_t["rsi_14"] = (df_t["ret_1d"].rolling(14).mean() / 
                          (df_t["ret_1d"].rolling(14).std() + 1e-6)).clip(-3, 3) * 50 + 50
        df_t["target"] = df_t["ret_1d"].shift(-1)
        feat.append(df_t[["ticker","date","ret_1d","rsi_14","target"]])
    features = pd.concat(feat).dropna().reset_index(drop=True)
    features_path = sample_dir / "features_sample.parquet"
    features.to_parquet(features_path, index=False)

    print("Wrote:")
    print(" -", prices_path)
    print(" -", features_path)

    if args.mirror_processed:
        dst_prices = processed_dir / "prices.parquet"
        dst_features = processed_dir / "features.parquet"
        for src, dst in [(prices_path, dst_prices), (features_path, dst_features)]:
            if dst.exists() and not args.overwrite:
                print(f"[skip] {dst} already exists (use --overwrite to replace)")
            else:
                shutil.copy(src, dst)
                print(f"[mirror] {src} -> {dst}")

if __name__ == "__main__":
    main()
