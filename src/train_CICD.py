import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
TARGET_COLUMN = "TX_FRAUD"
N_TRIALS = 5
N_SPLITS = 5
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
    return X, y


def objective(trial, X, y):
    with mlflow.start_run(nested=True):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
        }

        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        aucs = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)
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
    with mlflow.start_run(run_name="Training"):
        X, y = load_data()
        X, _, y, _ = train_test_split(X, y, test_size=0.9, stratify=y, random_state=RANDOM_STATE)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train),
            n_trials=N_TRIALS,
        )

        best_params = study.best_params
        best_params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": RANDOM_STATE,
                "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
                "tree_method": "hist",
            }
        )

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_auc", study.best_value)

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        y_proba_test = model.predict_proba(X_test)[:, 1]
        threshold, f1 = find_best_threshold(y_test, y_proba_test)

        y_pred = (y_proba_test >= threshold).astype(int)

        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
        mlflow.log_param("decision_threshold", float(threshold))

        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model, max_num_features=15)
        plt.tight_layout()
        plt.savefig("feature_importance_xgb.png")
        plt.close()

        mlflow.log_artifact("feature_importance_xgb.png")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="Fraud_XGBoost_Model",
        )

        print("Training completed")


if __name__ == "__main__":
    main()
