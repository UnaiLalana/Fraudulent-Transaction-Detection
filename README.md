# Fraudulent Transaction Detection

### Authors: Unai Lalana Morales, Lander Jimenez Nadales & Eneko Isturitz Sesma
#
This project is a complete pipeline for detecting fraudulent transactions using Machine Learning. It includes data ingestion, model training with hyperparameter optimization, experiment tracking, model serialization, and a REST API for real-time inference.

## 📊 Dataset
The project uses the [Fraud Transaction Detection](https://www.kaggle.com/datasets/sanskar457/fraud-transaction-detection) dataset from Kaggle.
* **Input Features:** Transaction amount, time, and custom engineered features (e.g., transaction hour, day of week).
* **Target:** `TX_FRAUD` (Binary: 0 for legitimate, 1 for fraudulent).

## 🛠 Tech Stack
* **Language:** Python 3.11+
* **Package Manager:** [uv](https://github.com/astral-sh/uv)
* **ML Core:** [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/)
* **Experiment Tracking:** [MLflow](https://mlflow.org/)
* **Optimization:** [Optuna](https://optuna.org/)
* **API:** [FastAPI](https://fastapi.tiangolo.com/)
* **Monitoring:** [Prometheus Client](https://github.com/prometheus/client_python)

## 🚀 Installation

Ensure you have Python 3.11 or higher installed. This project uses `uv` for dependency management.

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Fraudulent-Transaction-Detection
   ```

2. **Install dependencies:**
   You can use the provided Makefile for convenience.
   ```bash
   make install
   ```
   Or manually:
   ```bash
   pip install uv
   uv sync
   ```

## 🔄 Workflow

### 1. Download Data
Download the dataset from Kaggle automatically.
```bash
uv run python src/dataset_download.py
```
*Note: This script requires internet access to fetch data from Kaggle.*

### 2. Train Model
Train the XGBoost model using Optuna for hyperparameter tuning. Experiments are logged to MLflow.
```bash
uv run python src/train.py
```
*   **Outputs:** Logs metrics (AUC, F1, Precision, Recall) to MLflow and locally saves `feature_importance.png`.
*   **Artifacts:** The trained model is registered in MLflow under `Fraud_XGBoost_Model`.

### 3. Serialize Model
Extract the best model and decision threshold from MLflow and save them to the `artifacts/` directory for the API to use.
```bash
uv run python src/serialize.py
```
*   **Generates:** `artifacts/model.joblib` and `artifacts/threshold.json`.

### 4. Run API
Start the FastAPI server for real-time predictions.
```bash
uv run uvicorn api.api:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

## 📡 API Usage

### Health Check
```http
GET /
```

### Predict Fraud
**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "TX_AMOUNT": 150.50,
  "TX_TIME_SECONDS": 3600,
  "TX_HOUR": 14,
  "TX_DAY": 12,
  "TX_DAYOFWEEK": 5,
  "TX_MONTH": 1,
  "TX_IS_WEEKEND": 1
}
```

**Response:**
```json
{
  "fraud_probability": 0.85,
  "fraud_prediction": 1,
  "decision_threshold": 0.45
}
```

### Metrics
Prometheus metrics are exposed for monitoring.
```http
GET /metrics
```

## 🧪 Development

### Running Tests
Run basic tests using pytest.
```bash
make test
```

### Code Formatting & Linting
Format code with `black` and lint with `pylint`.
```bash
make format
make lint
```
