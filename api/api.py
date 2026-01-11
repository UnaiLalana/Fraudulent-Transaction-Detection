from fastapi import FastAPI, Response
from prometheus_client import (
    Counter,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from src.inference import predict
from pydantic import BaseModel
from collections import defaultdict, deque
import json
import numpy as np

app = FastAPI(title="Fraud Detection API")
app.state.requests_since_start = 0
REQUEST_COUNT = Counter(
    "api_request_count",
    "Number of API requests"
)

FRAUD_PROB_SUM = Gauge(
    "fraud_probability_sum",
    "Sum of fraud probabilities"
)

DATA_DRIFT_PSI = Gauge(
    "data_drift_psi",
    "Population Stability Index",
    ["feature"]
)

with open("artifacts/baseline_stats.json", "r", encoding="utf-8") as f:
    BASELINE = json.load(f)
BUFFER_SIZE = 20
MIN_PSI_SAMPLES = 3            
REQUESTS_FOR_PSI = 10

live_buffer = defaultdict(
    lambda: deque(maxlen=BUFFER_SIZE)
)

for f in BASELINE.keys():
    DATA_DRIFT_PSI.labels(feature=f).set(0.0)
class Transaction(BaseModel):
    TX_AMOUNT: float
    TX_TIME_SECONDS: int
    TX_HOUR: int
    TX_DAY: int
    TX_DAYOFWEEK: int
    TX_MONTH: int
    TX_IS_WEEKEND: int

def calculate_psi(expected_counts, bins, actual_values):
    actual_counts, _ = np.histogram(actual_values, bins=bins)

    actual_perc = actual_counts / max(len(actual_values), 1)
    expected_perc = np.array(expected_counts) / sum(expected_counts)

    psi = np.sum(
        (actual_perc - expected_perc) *
        np.log((actual_perc + 1e-6) / (expected_perc + 1e-6))
    )

    return float(psi)

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is alive!"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    app.state.requests_since_start += 1
    REQUEST_COUNT.inc()

    payload = transaction.dict()

    for feature in BASELINE.keys():
        live_buffer[feature].append(payload[feature])
    result = predict(payload)
    FRAUD_PROB_SUM.inc(result["fraud_probability"])

    if app.state.requests_since_start % REQUESTS_FOR_PSI == 0:
        for feature, buffer in live_buffer.items():
            if len(buffer) >= MIN_PSI_SAMPLES:
                psi = calculate_psi(
                    BASELINE[feature]["counts"],
                    BASELINE[feature]["bins"],
                    list(buffer)
                )
                DATA_DRIFT_PSI.labels(feature=feature).set(psi)

    return result

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
