# src/quant_trader/simulation/exact.py
from __future__ import annotations
import math
from typing import Set, List, Optional
import numpy as np
import pandas as pd

def _topk_by_pred(preds_for_day: pd.DataFrame, k: int) -> List[str]:
    if preds_for_day.empty:
        return []
    return (
        preds_for_day.sort_values("y_pred", ascending=False)
        .head(max(k, 0))["ticker"]
        .tolist()
    )

def run_exact_long_only_topk(
    preds: pd.DataFrame,
    k: int = 5,
    initial_capital: float = 100_000.0,
    slippage_bps: float = 5.0,
    commission_per_trade: float = 0.0,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Exact daily rebalance long-only Top-K (optional threshold on y_pred).
    Expects columns: ['ticker','date','y_true','y_pred'] with y_true = next-day *log* return.
    Returns: ['date','ret_port','equity','positions','turnover','cost_value'].
    """
    if preds.empty:
        return pd.DataFrame(columns=["date","ret_port","equity","positions","turnover","cost_value"])

    df = preds.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["y_true","y_pred"]).sort_values(["date","ticker"])

    wealth = float(initial_capital)
    prev_positions: Set[str] = set()
    rows = []

    # 🚫 No equality filter — iterate by group to avoid tz/time mismatches
    for d, day in df.groupby("date", sort=True):
        # Apply threshold if provided
        if threshold is not None:
            day = day[day["y_pred"] > threshold]

        # Pick Top-K and build weights
        tickers = _topk_by_pred(day, k)
        positions = set(tickers)
        n = len(tickers)
        target_w = {t: 1.0 / n for t in tickers} if n > 0 else {}

        # Realized simple return from next-period log returns
        if n == 0:
            port_ret_gross = 0.0
        else:
            simple_ret = np.exp(day.loc[day["ticker"].isin(positions), "y_true"].values) - 1.0
            port_ret_gross = float(simple_ret.mean()) if simple_ret.size else 0.0

        # Turnover (L1/2)
        old_w = {t: (1.0 / len(prev_positions)) if prev_positions else 0.0 for t in prev_positions}
        all_names = prev_positions | positions
        l1 = sum(abs(target_w.get(t, 0.0) - old_w.get(t, 0.0)) for t in all_names)
        turnover = 0.5 * l1

        # Costs
        trade_notional = wealth * turnover
        slippage_cost = trade_notional * (slippage_bps / 10_000.0)
        trades = sum(1 for t in all_names if target_w.get(t, 0.0) != old_w.get(t, 0.0))
        commission_cost = trades * commission_per_trade
        total_cost = slippage_cost + commission_cost

        # Apply costs then gross return
        wealth_after_costs = max(wealth - total_cost, 0.0)
        wealth_next = wealth_after_costs * (1.0 + port_ret_gross)

        ret_port = math.log(wealth_next / wealth) if wealth > 0 else 0.0
        rows.append({
            "date": d,
            "ret_port": ret_port,
            "equity": wealth_next,
            "positions": ",".join(sorted(positions)),
            "turnover": turnover,
            "cost_value": total_cost,
        })

        wealth = wealth_next
        prev_positions = positions

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

