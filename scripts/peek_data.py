import pandas as pd
p = "data/processed/prices.parquet"
df = pd.read_parquet(p)
print("Path:", p)
print("Rows, Cols:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(10))
