from pathlib import Path
import sys, pandas as pd
import matplotlib.pyplot as plt

CSV = Path("outputs/backtests/equity_curve.csv")
OUT = Path("docs/example_equity_curve.png")

if not CSV.exists():
    print(f"ERROR: {CSV} not found. Run `python scripts/simulate.py` first.", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(CSV)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["equity"])
plt.title("Strategy Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=160)
print(f"Saved -> {OUT}")
