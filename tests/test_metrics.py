import sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from src.quant_trader.simulation.metrics import summarize  # noqa: E402


def test_summarize_has_expected_fields():
    # build a tiny returns series
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    rng = np.random.default_rng(0)
    # pretend these are *log* daily portfolio returns
    rets = pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates, name="ret_port")
    df = pd.DataFrame({"date": dates, "ret_port": rets.values})

    m = summarize(df)
    for key in ("CAGR", "Sharpe", "MaxDD", "N"):
        assert key in m
        assert m["N"] == len(df)
