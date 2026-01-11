# scripts/simulate_requests.py
import requests
import pandas as pd
import time
import random
from tqdm import tqdm
import urllib3

# Suprime los warnings de HTTPS
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

df = pd.read_csv("data/data_drift.csv")

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

API_URL = "https://fraudulent-transaction-prediction-latest.onrender.com/predict"

for _, row in tqdm(df.iterrows(), total=len(df), desc="Sending requests"):
    try:
        requests.post(API_URL, json=row.to_dict(), verify=False)
    except Exception as e:
        print(f"Error enviando fila: {e}")
    time.sleep(15)