# scripts/report.py
import pathlib, pandas as pd, matplotlib.pyplot as plt
from src.quant_trader.simulation.metrics import summarize

def collect_backtests(folder: pathlib.Path, prefix: str):
    files = sorted(folder.glob(f"{prefix}*.parquet"), key=lambda x: x.stat().st_mtime)
    return files

def plot_all(backtests, out_path: pathlib.Path, title: str):
    plt.figure(figsize=(10,6))
    for f in backtests:
        df = pd.read_parquet(f)
        if "equity" not in df.columns:
            continue
        plt.plot(df["date"], df["equity"], lw=1.8, label=f.stem)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] Saved comparison → {out_path}")

if __name__ == "__main__":
    out_bt = pathlib.Path("outputs/backtests")
    out_plots = pathlib.Path("outputs/plots")
    out_plots.mkdir(parents=True, exist_ok=True)

    if not out_bt.exists():
        print("[report] No backtest directory found.")
        raise SystemExit(0)

    # === Vectorized Backtests ===
    vec_files = collect_backtests(out_bt, "vec_")
    if vec_files:
        print(f"[report] Found {len(vec_files)} vectorized backtests")
        for f in vec_files:
            df = pd.read_parquet(f)
            print(f"→ {f.name}")
            print(summarize(df))
        plot_all(vec_files, out_plots / "vec_comparison.png", "Vectorized Backtests")
    else:
        print("[report] No vectorized backtests found.")

    # === Exact Backtests ===
    ex_files = collect_backtests(out_bt, "exact_")
    if ex_files:
        print(f"[report] Found {len(ex_files)} exact backtests")
        for f in ex_files:
            df = pd.read_parquet(f)
            print(f"→ {f.name}")
            print(summarize(df.rename(columns={"equity":"_"}).assign(ret_port=df["ret_port"])))
        plot_all(ex_files, out_plots / "exact_comparison.png", "Exact Backtests")
    else:
        print("[report] No exact backtests found.")

