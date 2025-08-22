# scripts/run_pipeline.py

import sys, os, argparse, pathlib
repo = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo))

from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
load_dotenv()  # loads variables from .env into os.environ

from src.quant_trader.utils.config import load_config
from src.quant_trader.io.loaders import fetch_all
from src.quant_trader.features.feature_set import build_feature_matrix
from src.quant_trader.modeling.baselines import run_baseline
from src.quant_trader.simulation.vectorized import long_only_topk
from src.quant_trader.simulation.exact import run_exact_long_only_topk
from src.quant_trader.simulation.metrics import summarize


def main(cfg_path: str, k: int, threshold: float | None, file_mode: bool):
    cfg = load_config(cfg_path)

    proc_dir = Path("data/processed")
    out_pred = Path("outputs/predictions")
    out_bt = Path("outputs/backtests")
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)
    out_bt.mkdir(parents=True, exist_ok=True)

    # 1) DATA (Parquet + incremental loads)
    prices_path = proc_dir / "prices.parquet"

    if file_mode and prices_path.exists():
        df = pd.read_parquet(prices_path)
        print(f"[data] using existing {prices_path} rows={len(df)}")
    else:
        df_new = fetch_all(cfg)
        if prices_path.exists():
            df_old = pd.read_parquet(prices_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
        else:
            df = df_new

        df.to_parquet(prices_path, index=False)
        print(f"[data] updated {prices_path} rows={len(df)}")

    # 2) FEATURES
    X, y, meta = build_feature_matrix(df, cfg)
    feat = X.copy(); feat["target"] = y; feat = feat.reset_index()
    (proc_dir / "features.parquet").write_bytes(feat.to_parquet(index=False) or b"")
    print(f"[features] saved {proc_dir/'features.parquet'} rows={len(feat)}")

    # 3) MODEL (baseline DT) -> predictions
    metrics = run_baseline(
        features_path=str(proc_dir / "features.parquet"),
        out_path=str(out_pred / "baseline.parquet"),
        max_depth=3,
        test_quantile=0.80,
        random_state=cfg.get("project", {}).get("seed", 42),
    )
    print("[model]", metrics)

    # 4) SIMS (vectorized + exact)
    preds = pd.read_parquet(out_pred / "baseline.parquet")

    vec = long_only_topk(preds, k=k, threshold=threshold)
    vec_path = out_bt / f"vec_k{k}_thr{('none' if threshold is None else f'{threshold:.0e}')}.parquet"
    vec.to_parquet(vec_path, index=False)
    print("[sim vec]", summarize(vec), "->", vec_path)

    ex = run_exact_long_only_topk(
        preds, k=k, initial_capital=100_000.0,
        slippage_bps=5.0, commission_per_trade=0.0,
        threshold=threshold,
    )
    ex_path = out_bt / f"exact_k{k}_thr{('none' if threshold is None else f'{threshold:.0e}')}.parquet"
    ex.to_parquet(ex_path, index=False)
    print("[sim exact]", summarize(ex.rename(columns={"equity":"_"}).assign(ret_port=ex["ret_port"])), "->", ex_path)

    print("Pipeline complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--file-mode", action="store_true", help="Reuse data/processed/*.parquet (no downloads)")
    args = ap.parse_args()
    main(args.config, args.k, args.threshold, args.file_mode)
