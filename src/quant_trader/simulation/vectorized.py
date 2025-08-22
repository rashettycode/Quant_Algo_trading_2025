# src/quant_trader/simulation/vectorized.py
from __future__ import annotations
import numpy as np
import pandas as pd

def long_only_topk(
    preds: pd.DataFrame,
    k: int = 5,
    threshold: float | None = None,   # NEW optional arg
) -> pd.DataFrame:
    """
    Vectorized Long-Only Top-K by predicted return.
    Expects preds with columns: ['ticker','date','y_true','y_pred'] where:
      - y_pred: predicted next-day log return
      - y_true: realized next-day log return (the target you created)

    Parameters
    ----------
    preds : pd.DataFrame
        DataFrame with ['ticker','date','y_true','y_pred']
    k : int, default=5
        Max number of assets to hold per day.
    threshold : float, optional
        Only include assets where predicted return > threshold.

    Returns
    -------
    pd.DataFrame
        ['date','ret_port'] daily portfolio log returns.
    """
    if preds.empty:
        return pd.DataFrame(columns=["date", "ret_port"])

    df = preds.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["y_true", "y_pred"])

    # filter by threshold
    if threshold is not None:
        df = df[df["y_pred"] > threshold]

    # rank by prediction within each date (higher is better)
    df["rank"] = df.groupby("date")["y_pred"].rank(method="first", ascending=False)

    # select top-k
    sel = df[df["rank"] <= k].copy()

    # equal-weight mean of realized returns per day
    port = (
        sel.groupby("date", as_index=False)["y_true"]
           .mean()
           .rename(columns={"y_true": "ret_port"})
           .sort_values("date")
    )

    return port

def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """
    Compound log returns into an equity curve.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns of the portfolio.
    initial : float, default=1.0
        Starting equity value.

    Returns
    -------
    pd.Series
        Equity curve values over time.
    """
    if returns.empty:
        return pd.Series([], dtype=float)
    return initial * np.exp(returns.cumsum())
