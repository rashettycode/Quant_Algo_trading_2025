import pandas as pd
from pathlib import Path

def read_parquet_or_empty(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)

def upsert_parquet(df_new: pd.DataFrame, path: str, key=["ticker", "date"]) -> pd.DataFrame:
    """Load existing parquet, append new rows, drop duplicates."""
    df_old = read_parquet_or_empty(path)
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=key).sort_values(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(path, index=False)
    return df_all
