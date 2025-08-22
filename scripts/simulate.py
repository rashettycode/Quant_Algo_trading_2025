# scripts/simulate.py
import sys, argparse, pathlib, pandas as pd
repo = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo))

from pathlib import Path
from src.quant_trader.utils.config import load_config
from src.quant_trader.simulation.vectorized import long_only_topk
from src.quant_trader.simulation.exact import run_exact_long_only_topk
from src.quant_trader.simulation.metrics import summarize

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_pred = Path("outputs/predictions")
    out_bt = Path("outputs/backtests"); out_bt.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(out_pred / "baseline.parquet")

    # Vectorized
    vec = long_only_topk(preds, k=args.k, threshold=args.threshold)
    vec_path = out_bt / f"vec_k{args.k}_thr{('none' if args.threshold is None else f'{args.threshold:.0e}')}.parquet"
    vec.to_parquet(vec_path, index=False)
    print("[sim vec]", summarize(vec), "->", vec_path)

    # Exact
    ex = run_exact_long_only_topk(
        preds, k=args.k,
        initial_capital=100_000.0,
        slippage_bps=5.0, commission_per_trade=0.0,
        threshold=args.threshold,
    )
    ex_path = out_bt / f"exact_k{args.k}_thr{('none' if args.threshold is None else f'{args.threshold:.0e}')}.parquet"
    ex.to_parquet(ex_path, index=False)
    print("[sim exact]", summarize(ex.rename(columns={"equity":"_"}).assign(ret_port=ex["ret_port"])), "->", ex_path)
