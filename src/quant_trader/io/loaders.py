# --- Add near the top of the file (after other imports) ---
import os
import time
import pandas as pd
from typing import Optional

#  optional dependencies once (avoids Pylance "unresolved import" noise)
try:
    from fredapi import Fred  # type: ignore
    _HAVE_FREDAPI = True
except Exception:
    _HAVE_FREDAPI = False

try:
    from pandas_datareader import data as pdr  # type: ignore
    _HAVE_PDR = True
except Exception:
    _HAVE_PDR = False


def _download_alpha_vantage(tickers: list[str], api_key: str, outputsize: str = "compact") -> pd.DataFrame:
    """
    TIME_SERIES_DAILY_ADJUSTED for each ticker.
    Free keys are rate-limited (~5 calls/min). We sleep between calls.
    Returns tidy long: ['ticker','date','open','high','low','close','adj_close','volume']
    """
    import requests  # local import keeps optional dep warnings down

    rows: list[dict] = []
    for i, t in enumerate(tickers):
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": t,
            "outputsize": outputsize,
            "datatype": "json",
            "apikey": api_key,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()

        # Gracefully handle rate-limit or errors
        if "Note" in js or "Error Message" in js:
            # You can log js.get("Note") here if you want visibility
            if i < len(tickers) - 1:
                time.sleep(12)
            continue

        ts = js.get("Time Series (Daily)", {}) or {}
        for d, vals in ts.items():
            rows.append(
                {
                    "ticker": t,
                    "date": pd.to_datetime(d),
                    "open": float(vals["1. open"]),
                    "high": float(vals["2. high"]),
                    "low": float(vals["3. low"]),
                    "close": float(vals["4. close"]),
                    "adj_close": float(vals.get("5. adjusted close", vals["4. close"])),
                    "volume": float(vals["6. volume"]),
                }
            )

        # be nice to the API
        if i < len(tickers) - 1:
            time.sleep(12)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def _download_fred(series_ids: list[str], api_key: Optional[str], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """
    Fetch macro series from FRED. Prefer fredapi (with API key); fallback to pandas_datareader if needed.
    Returns tidy long: ['series','date','value'] with one row per series_id per date.
    """
    out: list[pd.DataFrame] = []

    # Try fredapi first
    if _HAVE_FREDAPI:
        try:
            fred = Fred(api_key=api_key)
            for sid in series_ids:
                s = fred.get_series(series_id=sid, observation_start=start, observation_end=end)
                df = s.to_frame(name="value").reset_index()
                # Normalize date col name
                if "index" in df.columns:
                    df = df.rename(columns={"index": "date"})
                elif "DATE" in df.columns:
                    df = df.rename(columns={"DATE": "date"})
                df["series"] = sid
                out.append(df)
        except Exception:
            # fall through to pdr
            pass

    # Fallback: pandas_datareader
    if not out and _HAVE_PDR:
        for sid in series_ids:
            df = pdr.DataReader(sid, "fred", start=start, end=end)
            # df index is DATE; column name = sid
            df = df.rename_axis("date").reset_index().rename(columns={sid: "value"})
            df["series"] = sid
            out.append(df)

    if not out:
        # No provider available or nothing fetched
        return pd.DataFrame(columns=["series", "date", "value"])

    df_all = pd.concat(out, ignore_index=True)

    # Enforce types & normalize
    if "date" not in df_all.columns:
        # last resort: try common date labels
        for candidate in ("DATE", "Index", "index"):
            if candidate in df_all.columns:
                df_all = df_all.rename(columns={candidate: "date"})
                break

    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["value"] = pd.to_numeric(df_all["value"], errors="coerce")
    return df_all.sort_values(["series", "date"]).reset_index(drop=True)


def merge_prices_and_macro(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join macro series onto prices by date (broadcast macro across tickers).
    - Forward-fill macro daily to align with business days.
    - Returns prices with extra macro columns (wide by series).
    """
    if macro.empty:
        return prices

    mwide = macro.pivot(index="date", columns="series", values="value").sort_index().ffill()

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    merged = prices.merge(mwide.reset_index(), on="date", how="left")

    # final ffill for any leading NaNs
    for c in mwide.columns:
        merged[c] = merged[c].ffill()

    return merged


def fetch_all(cfg: dict) -> pd.DataFrame:
    """
    Unified data fetcher:
      - Prices from Yahoo (default) or Alpha Vantage (if enabled)
      - Optional FRED macro merge (if enabled)
    """
    start = cfg.get("data", {}).get("start_date")
    end = cfg.get("data", {}).get("end_date")
    tickers: list[str] = cfg.get("data", {}).get("tickers", [])

    use_yahoo = cfg.get("sources", {}).get("yahoo", {}).get("use", True)
    use_av = cfg.get("sources", {}).get("alpha_vantage", {}).get("use", False)
    use_fred = cfg.get("sources", {}).get("fred", {}).get("use", False)

    # 1) Prices
    if use_yahoo and tickers:
        # assumes your existing _download_yahoo(...) is defined elsewhere in this file
        df_prices = _download_yahoo(tickers, start, end)  # type: ignore[name-defined]
    elif use_av and tickers:
        av_cfg = cfg.get("sources", {}).get("alpha_vantage", {})
        api_key = os.getenv(av_cfg.get("api_key_env", "ALPHAVANTAGE_API_KEY"), "")
        if not api_key:
            raise RuntimeError("Alpha Vantage selected but API key not found in env.")
        outputsize = av_cfg.get("outputsize", "compact")
        df_prices = _download_alpha_vantage(tickers, api_key, outputsize=outputsize)
    else:
        df_prices = pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"])

    # 2) Macro (FRED)
    if use_fred:
        fred_cfg = cfg.get("sources", {}).get("fred", {})
        fred_key = os.getenv(fred_cfg.get("api_key_env", "FRED_API_KEY"), None)
        series_ids = fred_cfg.get("series", []) or []
        df_macro = _download_fred(series_ids, fred_key, start, end) if series_ids else pd.DataFrame()
        df_prices = merge_prices_and_macro(df_prices, df_macro)

    return df_prices


