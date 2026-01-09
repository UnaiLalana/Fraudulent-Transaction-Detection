import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# --------------------------
# GPU device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
# --------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
TARGET_COLUMN = "TX_FRAUD"
N_TRIALS = 1
N_SPLITS = 3
RANDOM_STATE = 42

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_experiment("Fraud_Detection")


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
    df["TX_DAY"] = df["TX_DATETIME"].dt.day
    df["TX_DAYOFWEEK"] = df["TX_DATETIME"].dt.dayofweek
    df["TX_MONTH"] = df["TX_DATETIME"].dt.month
    df["TX_IS_WEEKEND"] = df["TX_DAYOFWEEK"].isin([5, 6]).astype(int)

    X = df.drop(
        columns=[
            TARGET_COLUMN,
            "TRANSACTION_ID",
            "Unnamed: 0",
            "TX_FRAUD_SCENARIO",
            "TX_DATETIME",
            "TERMINAL_ID",
            "CUSTOMER_ID",
            "TX_TIME_DAYS",
        ]
    )

    y = df[TARGET_COLUMN]
    return X.values, y.values, X.columns


def objective(trial, X, y):
    with mlflow.start_run(nested=True):
        # --- parametros Optuna ---
        params = {
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_a": trial.suggest_int("n_a", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 8),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
            "optimizer_params": {
                "lr": trial.suggest_float("lr", 1e-3, 0.05, log=True)
            },
            "mask_type": "entmax",
            "seed": RANDOM_STATE,
        }

        mlflow.log_params(params)

        cv = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )

        aucs = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = TabNetClassifier(**params, device_name=DEVICE)

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=["auc"],
                max_epochs=1,
                patience=3,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
            )

            y_proba = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, y_proba))

        mean_auc = np.mean(aucs)
        mlflow.log_metric("cv_roc_auc", mean_auc)

        return mean_auc


def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def main():
    with mlflow.start_run(run_name="Training_TabNet"):
        X, y, feature_names = load_data()

        # Usar solo 10% del dataset
        X, _, y, _ = train_test_split(
            X,
            y,
            test_size=0.9,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        # ----------------------
        # Optuna search
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train),
            n_trials=N_TRIALS,
        )

        best_params = study.best_params
        # --- mover lr dentro de optimizer_params ---
        lr = best_params.pop("lr")
        best_params["optimizer_params"] = {"lr": lr}
        best_params.update({"mask_type": "entmax", "seed": RANDOM_STATE})
        # ----------------------

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_auc", study.best_value)

        # ----------------------
        # Modelo final
        model = TabNetClassifier(**best_params, device_name=DEVICE)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=["auc"],
            max_epochs=5,
            patience=5,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )

        # Predicciones
        y_proba_test = model.predict_proba(X_test)[:, 1]
        threshold, f1 = find_best_threshold(y_test, y_proba_test)
        y_pred = (y_proba_test >= threshold).astype(int)

        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
        mlflow.log_param("decision_threshold", float(threshold))

        # Feature importance
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:15]

        plt.figure(figsize=(10, 6))
        plt.barh(
            [feature_names[i] for i in idx][::-1],
            importances[idx][::-1],
        )
        plt.title("Top 15 Feature Importances (TabNet)")
        plt.tight_layout()
        plt.savefig("feature_importance_tabnet.png")
        plt.close()

        mlflow.log_artifact("feature_importance_tabnet.png")

        print("Training completed with TabNet")

if __name__ == "__main__":
    main()
