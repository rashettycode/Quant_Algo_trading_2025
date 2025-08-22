import os
import shutil
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def synthetic_parquet():
    """
    Create a small synthetic prices.parquet so the pipeline can run in --file-mode
    without any network calls. Cleans outputs to keep CI tidy.
    """
    repo = Path(__file__).resolve().parents[1]
    proc = repo / "data" / "processed"
    out = repo / "outputs"

    # Fresh dirs
    proc.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "backtests").mkdir(parents=True, exist_ok=True)

    # Tiny price panel: 40 business days for 3 tickers
    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    tickers = ["AAPL", "MSFT", "SPY"]
    rows = []
    rng = np.random.default_rng(42)
    for t in tickers:
        price = 100.0
        for d in dates:
            price *= 1 + rng.normal(0.0005, 0.01)
            rows.append(
                {
                    "ticker": t,
                    "date": d,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(proc / "prices.parquet", index=False)

    # yield to tests; clean up after the whole session
    yield

    # Cleanup (optional in local dev, helpful in CI)
    try:
        shutil.rmtree(out, ignore_errors=True)
        # Keep processed parquet for developer convenience
    except Exception:
        pass
