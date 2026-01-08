import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.json"

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    threshold = float(json.load(f)["decision_threshold"])


def predict(input_data: dict) -> dict:
    """
    input_data: dict {feature_name: value}
    """

    df = pd.DataFrame([input_data])

    proba = model.predict_proba(df)[:, 1][0]

    prediction = int(proba >= threshold)

    return {
        "fraud_probability": float(proba),
        "fraud_prediction": prediction,
        "decision_threshold": threshold,
    }
