# scripts/download_data.py
import sys, argparse, pathlib, pandas as pd
repo = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo))

from pathlib import Path
from src.quant_trader.utils.config import load_config
from src.quant_trader.io.loaders import fetch_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--file-mode", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc_dir = Path("data/processed"); proc_dir.mkdir(parents=True, exist_ok=True)
    prices_path = proc_dir / "prices.parquet"

    if args.file_mode and prices_path.exists():
        df = pd.read_parquet(prices_path)
        print(f"[data] using existing {prices_path}, rows={len(df)}")
    else:
        df = fetch_all(cfg)
        df.to_parquet(prices_path, index=False)
        print(f"[data] saved {prices_path}, rows={len(df)}")



