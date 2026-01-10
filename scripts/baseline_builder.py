# scripts/build_baseline.py
import json
import pandas as pd

df = pd.read_csv("data/train.csv")


df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
df["TX_DAY"] = df["TX_DATETIME"].dt.day
df["TX_DAYOFWEEK"] = df["TX_DATETIME"].dt.dayofweek
df["TX_MONTH"] = df["TX_DATETIME"].dt.month
df["TX_IS_WEEKEND"] = df["TX_DAYOFWEEK"].isin([5, 6]).astype(int)

df = df.drop(
    columns=[
        "TX_FRAUD",
        "TRANSACTION_ID",
        "Unnamed: 0",
        "TX_FRAUD_SCENARIO",
        "TX_DATETIME",
        "TERMINAL_ID",
        "CUSTOMER_ID",
        "TX_TIME_DAYS",
    ]
)

baseline = {}

for col in df.columns:
    baseline[col] = {
        "mean": df.loc[:, col].mean(),
        "std": df.loc[:, col].std(),
        "quantiles": df.loc[:, col].quantile([0.1,0.5,0.9]).tolist()
    }

with open("artifacts/baseline_stats.json", "w") as f:
    json.dump(baseline, f)
