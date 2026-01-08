import gradio as gr
import requests
from datetime import datetime

API_URL = "https://fraudulent-transaction-prediction-latest.onrender.com"

def predict_from_ui(amount, datetime_value, elapsed_seconds):
    if datetime_value is None or elapsed_seconds == 0:
        return "Fill the form with data", 0.0

    if isinstance(datetime_value, float):
        dt = datetime.fromtimestamp(datetime_value)
    else:
        dt = datetime_value

    payload = {
        "TX_AMOUNT": amount,
        "TX_HOUR": dt.hour,
        "TX_DAY": dt.day,
        "TX_DAYOFWEEK": dt.weekday(),
        "TX_MONTH": dt.month,
        "TX_IS_WEEKEND": 1 if dt.weekday() >= 5 else 0,
        "TX_TIME_SECONDS": elapsed_seconds
    }

    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        r.raise_for_status()
        result = r.json()
    except Exception as e:
        return "Error calling the API", 0.0

    fraud_prob = result.get("fraud_probability", 0.0)
    fraud_label = "Fraud Detected" if fraud_prob > 0.5 else "Not Fraud Detected"

    return fraud_label, round(fraud_prob * 100, 2)

with gr.Blocks(title="Fraud Detection Demo") as demo:
    gr.Markdown("## 💳 Fraud Detection – Demo")
    gr.Markdown("Select the transaction amount, date, time and elapsed seconds:")

    amount = gr.Number(label="Transaction Amount", value=100.0)
    datetime_input = gr.DateTime(label="Transaction Date & Time", value=datetime.now())
    elapsed_seconds = gr.Slider(
    label="Elapsed Seconds",
    minimum=0,
    maximum=99999999,
    step=1,
    value=0
)
    
    submit = gr.Button("Predict Fraud")

    fraud_label_output = gr.Label(label="Prediction")
    confidence_output = gr.Number(label="Confidence (%)")

    submit.click(
        predict_from_ui,
        inputs=[amount, datetime_input, elapsed_seconds],
        outputs=[fraud_label_output, confidence_output]
    )

demo.launch()