# scripts/simulate_baseline.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path
import pandas as pd

from src.quant_trader.simulation.vectorized import long_only_topk, equity_curve
from src.quant_trader.simulation.metrics import summarize


def maybe_benchmark() -> pd.DataFrame:
    """
    Optional SPY benchmark (log returns) if present in prices.parquet.
    """
    p = Path("data/processed/prices.parquet")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    spy = df[df["ticker"] == "SPY"].sort_values("date").copy()
    if spy.empty:
        return pd.DataFrame()

    import numpy as np
    spy["ret_bench"] = np.log(1.0 + spy["close"].pct_change())
    return spy[["date", "ret_bench"]]


def main(k: int, threshold: float | None):
    preds_path = "outputs/predictions/baseline.parquet"
    out_dir = Path("outputs/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(preds_path)  # expects ['ticker','date','y_true','y_pred']

    # Vectorized Top-K (now with optional threshold)
    port = long_only_topk(preds, k=k, threshold=threshold)

    # Optional benchmark join (if SPY exists in your prices file)
    bench = maybe_benchmark()
    if not bench.empty:
        port = port.merge(bench, on="date", how="left")

    # Pick an output filename that reflects params
    if threshold is None:
        out_path = out_dir / f"vec_topk_k{k}.parquet"
    else:
        # sanitize threshold for filename (e.g., 0.001 -> 1e-03)
        out_path = out_dir / f"vec_topk_k{k}_thr{threshold:.0e}.parquet"

    port.to_parquet(out_path, index=False)

    # Metrics + equity preview
    m = summarize(port)
    print(f"Vectorized Top-K (k={k}, threshold={threshold}) Metrics:", m)
    print("Equity (last 5):")
    print(equity_curve(port['ret_port']).tail())
    print(f"Saved portfolio returns → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="max positions per day")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        nargs="?",
        help="only trade when y_pred > threshold (log-return units); default: no threshold",
    )
    args = ap.parse_args()

    main(k=args.k, threshold=args.threshold)

