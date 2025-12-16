import requests
import os
from pathlib import Path
import pathlib
import zipfile

def download_dataset(data_path: str):
    data_path = Path(data_path)
    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/sanskar457/fraud-transaction-detection"
    filename = "dataset.zip"
    csv_file = "dataset.csv"

    # Check if dataset already downloaded
    if os.path.exists(data_path):
        print("Data detected; Skipping download")
        return

    os.mkdir(data_path)

    # Download Data
    print("Downloading data...")
    req = requests.get(dataset_url, allow_redirects=True)
    if req.status_code != 200:
        print("Error fetchinf dataset")
        return
    open(data_path / filename, "wb+").write(req.content)
    print("Download complete!")

    # Extracting dataset
    zip_file = zipfile.ZipFile(data_path / filename, "r")
    with open(data_path / csv_file, "wb") as f:
        f.write(zip_file.read("Final Transactions.csv"))

if __name__ == "__main__":
    download_dataset("./data")
