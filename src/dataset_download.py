import requests
import os
from pathlib import Path
import zipfile
import pandas as pd


def download_dataset(data_path: str):
    data_path = Path(data_path)
    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/sanskar457/fraud-transaction-detection"
    filename = "dataset.zip"
    csv_file = "dataset.csv"

    if os.path.exists(data_path / csv_file):
        print("Data detected; Skipping download")
        return

    os.makedirs(data_path, exist_ok=True)

    print("Downloading data...")
    req = requests.get(dataset_url, allow_redirects=True, timeout=1000)
    if req.status_code != 200:
        raise RuntimeError("Error fetching dataset")

    with open(data_path / filename, "wb") as f:
        f.write(req.content)

    with zipfile.ZipFile(data_path / filename, "r") as zip_file:
        zip_file.extract("Final Transactions.csv", path=data_path)
        os.rename(data_path / "Final Transactions.csv", data_path / csv_file)

    print("Dataset downloaded and extracted")


def temporal_split(
    data_path: str,
    time_column: str,
    drift_ratio: float = 0.2,
):
    data_path = Path(data_path)
    df = pd.read_csv(data_path / "dataset.csv")

    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column).reset_index(drop=True)

    n_total = len(df)
    n_drift = int(n_total * drift_ratio)

    df_train = df.iloc[: n_total - n_drift]
    df_drift = df.iloc[n_total - n_drift :]

    df_train.to_csv(data_path / "train.csv", index=False)
    df_drift.to_csv(data_path / "data_drift.csv", index=False)

    print("Temporal split completed:")
    print(f"Train (CV inside): {len(df_train)} rows")
    print(f"Data drift (future): {len(df_drift)} rows")


if __name__ == "__main__":
    DATA_DIR = "./data"
    TIME_COLUMN = "TX_DATETIME"

    download_dataset(DATA_DIR)
    temporal_split(DATA_DIR, TIME_COLUMN)
