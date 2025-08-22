# scripts/predict.py
import pathlib, pandas as pd

if __name__ == "__main__":
    preds = pd.read_parquet(pathlib.Path("outputs/predictions/baseline.parquet"))
    print("[predict] predictions loaded, rows=", len(preds))
    print(preds.head())

