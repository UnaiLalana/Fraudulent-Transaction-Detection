# scripts/simulate_requests.py
import requests
import pandas as pd

df = pd.read_csv("data/drift_subset.csv")

for _, row in df.iterrows():
    requests.post(
        "https://fraudulent-transaction-prediction-latest.onrender.com/predict",
        json=row.to_dict()
    )
