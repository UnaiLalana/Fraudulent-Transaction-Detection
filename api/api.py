from fastapi import FastAPI
from prometheus_client import Counter, Gauge, generate_latest
from src.inference import predict
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Fraud Detection API")

REQUEST_COUNT = Counter("api_request_count", "Number of API requests")
FRAUD_PROB_SUM = Gauge("fraud_probability_sum", "Sum of fraud probabilities")


class Transaction(BaseModel):
    TX_AMOUNT: float
    TX_TIME_SECONDS: Optional[int] = None
    TX_HOUR: int
    TX_DAY: int
    TX_DAYOFWEEK: int
    TX_MONTH: int
    TX_IS_WEEKEND: int

    def complete_features(self):
        if self.TX_TIME_SECONDS is None:
            self.TX_TIME_SECONDS = self.TX_HOUR * 3600
    

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is alive!"}


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    REQUEST_COUNT.inc()

    transaction.complete_features()

    payload = transaction.dict()
    result = predict(payload)

    FRAUD_PROB_SUM.set(result["fraud_probability"])

    return result


@app.get("/metrics")
def metrics():
    return generate_latest()
