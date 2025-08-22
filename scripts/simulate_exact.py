# scripts/simulate_exact.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from pathlib import Path

from src.quant_trader.simulation.exact import run_exact_long_only_topk
from src.quant_trader.simulation.metrics import summarize

if __name__ == "__main__":
    preds_path = "outputs/predictions/baseline.parquet"
    out_path = "outputs/backtests/exact_topk.parquet"
    Path("outputs/backtests").mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(preds_path)  # ['ticker','date','y_true','y_pred']
    exact = run_exact_long_only_topk(
        preds,
        k=5,
        initial_capital=100_000.0,
        slippage_bps=5.0,
        commission_per_trade=0.0,
    )
    exact.to_parquet(out_path, index=False)

    m = summarize(exact.rename(columns={"equity":"_"}).assign(ret_port=exact["ret_port"]))
    print("Exact Long-Only Top-K Metrics:", m)
    print("Equity (last 5):")
    print(exact["equity"].tail())
    print(f"Saved exact sim → {out_path}")
