# src/quant_trader/features/feature_set.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _compute_rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothing (EMA with alpha=1/window).
    Returns a float Series in [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing (EMA) for average gain/loss
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def build_feature_matrix(df_prices: pd.DataFrame, cfg: dict):
    """
    Inputs:
      df_prices: tidy long OHLCV with columns:
                 ['ticker','date','open','high','low','close','adj_close','volume']
      cfg: (unused here, but kept for consistency)

    Output:
      X: DataFrame with index [ticker, date] and columns ['ret_1d','rsi_14']
      y: Series 'target' = next-day ret_1d (aligned with X index)
      meta: dict with 'index' (MultiIndex)
    """
    if df_prices is None or df_prices.empty:
        X = pd.DataFrame(columns=["ret_1d", "rsi_14"])
        y = pd.Series(name="target", dtype=float)
        meta = {"index": pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])}
        return X, y, meta

    df = (
        df_prices[["ticker", "date", "close"]]
        .dropna()
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .sort_values(["ticker", "date"])
        .copy()
    )

    def per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        # 1-day log return
        g["ret_1d"] = np.log(g["close"]).diff()
        # RSI(14) via Wilder's smoothing
        g["rsi_14"] = _compute_rsi_wilder(g["close"], window=14)
        # Target is next-day ret_1d
        g["target"] = g["ret_1d"].shift(-1)
        # Drop warmup rows (where RSI is NaN) and last row (target NaN)
        g = g.dropna(subset=["rsi_14"]).iloc[:-1] if len(g) > 0 else g
        return g

    out = df.groupby("ticker", group_keys=False).apply(per_ticker)
    out = out.set_index(["ticker", "date"]).sort_index()

    X = out[["ret_1d", "rsi_14"]].copy()
    y = out["target"].copy()
    meta = {"index": X.index}

    return X, y, meta
