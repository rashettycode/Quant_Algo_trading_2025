# src/quant_trader/simulation/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mu = returns.mean() * periods_per_year
    sigma = returns.std(ddof=1) * np.sqrt(periods_per_year)
    return 0.0 if sigma == 0 else float(mu / sigma)

def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    total_log_ret = returns.sum()
    years = len(returns) / periods_per_year
    return float(np.exp(total_log_ret / max(years, 1e-9)) - 1.0)

def summarize(port: pd.DataFrame) -> dict:
    """
    port: DataFrame with columns ['date','ret_port'].
    """
    if port.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "N": 0}

    eq = (1.0 * np.exp(port["ret_port"].cumsum())).rename("equity")
    return {
        "CAGR": cagr(port["ret_port"]),
        "Sharpe": sharpe_ratio(port["ret_port"]),
        "MaxDD": max_drawdown(eq),
        "N": int(len(port)),
    }
