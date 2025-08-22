# src/quant_trader/modeling/baselines.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_baseline(features_path: str = "data/processed/features.parquet",
                 out_path: str = "outputs/predictions/baseline.parquet",
                 max_depth: int = 3,
                 test_quantile: float = 0.8,
                 random_state: int = 42) -> dict:
    """
    Train a tiny DecisionTreeRegressor on ['ret_1d','rsi_14'] to predict 'target'.
    Splits by date using the given quantile (default: 80% train / 20% test).
    Saves test-set predictions to out_path.

    Returns a dict of simple metrics.
    """
    df = pd.read_parquet(features_path).dropna(subset=["ret_1d", "rsi_14", "target"])
    df["date"] = pd.to_datetime(df["date"])

    # time-based split
    cutoff = df["date"].quantile(test_quantile)
    train = df[df["date"] <= cutoff]
    test  = df[df["date"] >  cutoff]

    X_train, y_train = train[["ret_1d", "rsi_14"]], train["target"]
    X_test,  y_test  = test[["ret_1d", "rsi_14"]], test["target"]

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mse": float(mean_squared_error(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "cutoff": cutoff.isoformat(),
        "max_depth": max_depth,
    }

    # Save predictions for inspection/backtests later
    out = test[["ticker", "date"]].copy()
    out["y_true"] = y_test.values
    out["y_pred"] = preds
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    return metrics

