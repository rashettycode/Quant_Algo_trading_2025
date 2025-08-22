# scripts/train_models.py
import sys, argparse, pathlib, yaml
repo = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo))

from dotenv import load_dotenv
load_dotenv()  # loads variables from .env into os.environ

from pathlib import Path
from src.quant_trader.utils.config import load_config
from src.quant_trader.modeling.baselines import run_baseline


def load_model_params(models_yaml: str) -> dict:
    p = Path(models_yaml)
    if not p.exists():
        # Safe defaults if models.yaml is missing
        return {"max_depth": 3, "min_samples_leaf": 1}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    dt = (data.get("models", {}) or {}).get("decision_tree", {}) or {}
    return {
        "max_depth": int(dt.get("max_depth", 3)) if not isinstance(dt.get("max_depth"), list) else int(dt.get("max_depth")[0]),
        "min_samples_leaf": int(dt.get("min_samples_leaf", 1)) if not isinstance(dt.get("min_samples_leaf"), list) else int(dt.get("min_samples_leaf")[0]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--models", default="configs/models.yaml", help="Model config with tuned params")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc_dir = Path("data/processed")
    out_pred = Path("outputs/predictions"); out_pred.mkdir(parents=True, exist_ok=True)

    # Read tuned params (Optuna writes winners as scalars into configs/models.yaml)
    params = load_model_params(args.models)
    max_depth = params["max_depth"]
    min_samples_leaf = params["min_samples_leaf"]

    print(f"[train] Using DecisionTree params -> max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")

    metrics = run_baseline(
        features_path=str(proc_dir / "features.parquet"),
        out_path=str(out_pred / "baseline.parquet"),
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        test_quantile=0.80,
        random_state=cfg.get("project", {}).get("seed", 42),
    )
    print("[train]", metrics)

