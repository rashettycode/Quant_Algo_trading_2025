# scripts/tune_dt.py
import sys, argparse, pathlib, shutil
repo = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo))

import optuna
import pandas as pd
import yaml
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from src.quant_trader.modeling.datasets import make_splits
from src.quant_trader.utils.config import load_config


def objective(trial, feat_path: pathlib.Path):
    df = pd.read_parquet(feat_path)

    # minimal feature set; extend if you add more engineered features
    X = df[["ret_1d", "rsi_14"]]
    y = df["target"]
    meta = df[["ticker", "date"]]

    ds = make_splits(X, y, meta, cfg=None, test_quantile=0.80)

    # search space
    max_depth = trial.suggest_int("max_depth", 2, 12)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    model.fit(ds["X_train"], ds["y_train"])
    pred = model.predict(ds["X_test"])
    return mean_squared_error(ds["y_test"], pred)


def safe_load_yaml(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: pathlib.Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def update_models_yaml(models_yaml: pathlib.Path, best_params: dict):
    # backup first
    if models_yaml.exists():
        shutil.copyfile(models_yaml, models_yaml.with_suffix(".yaml.bak"))

    data = safe_load_yaml(models_yaml)

    # ensure structure
    data.setdefault("models", {})
    dt = data["models"].setdefault("decision_tree", {})
    dt["use"] = dt.get("use", True)
    # write best params
    dt["max_depth"] = int(best_params["max_depth"])
    dt["min_samples_leaf"] = int(best_params["min_samples_leaf"])

    write_yaml(models_yaml, data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--n-trials", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config(args.config)
    feat_path = pathlib.Path("data/processed/features.parquet")
    assert feat_path.exists(), "Run feature building first (e.g., `make run` once)."

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, feat_path), n_trials=args.n_trials)

    print("Best params:", study.best_params)
    print("Best MSE:", study.best_value)

    models_yaml = pathlib.Path(args.models)
    update_models_yaml(models_yaml, study.best_params)
    print(f"Updated {models_yaml} (backup at {models_yaml.with_suffix('.yaml.bak')})")
