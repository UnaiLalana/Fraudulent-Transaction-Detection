from fastapi import FastAPI, Response
import requests

app = FastAPI(title="Metrics Proxy")

EXTERNAL_METRICS_URL = "https://fraudulent-transaction-prediction-latest.onrender.com/metrics"

@app.get("/metrics")
def proxy_metrics():
    r = requests.get(EXTERNAL_METRICS_URL, verify=False)
    content = r.text

    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1]

    content = content.replace('\\n', '\n').replace('\\"', '"')

    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
