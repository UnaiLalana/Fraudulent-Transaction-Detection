import json
import os
from pathlib import Path

import joblib
import mlflow

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "Fraud_XGBoost_Model"
MODEL_STAGE = "None"

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
)

client = mlflow.tracking.MlflowClient()

model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.sklearn.load_model(model_uri)
joblib.dump(model, ARTIFACT_DIR / "model.joblib")

latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
run = client.get_run(latest.run_id)

threshold = float(run.data.params["decision_threshold"])

with open(ARTIFACT_DIR / "threshold.json", "w") as f:
    json.dump({"decision_threshold": threshold}, f)

print("Model and threshold correctly serialized")
