# scripts/make_fake_prices.py
import pandas as pd, numpy as np, pathlib

out = pathlib.Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

dates = pd.date_range("2024-01-02", periods=60, freq="B")
tickers = ["AAPL", "MSFT", "SPY"]

rows = []
rng = np.random.default_rng(42)
for t in tickers:
    price = 100.0
    for d in dates:
        price *= 1 + rng.normal(0.0005, 0.01)
        rows.append({
            "ticker": t,
            "date": d,
            "open": price * 0.99,
            "high": price * 1.01,
            "low": price * 0.98,
            "close": price,
            "adj_close": price,
            "volume": 1_000_000
        })

pd.DataFrame(rows).to_parquet(out / "prices.parquet", index=False)
print("Wrote", out / "prices.parquet")
