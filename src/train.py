import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import joblib
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
TARGET_COLUMN = "TX_FRAUD"
N_TRIALS = 5
N_SPLITS = 5
RANDOM_STATE = 42
mlflow.set_experiment("Fraud_Detection_XGBoost")


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
            "TX_FRAUD_SCENARIO",
            "TX_DATETIME"
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
            "tree_method": "hist"
        }

        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        aucs = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]

            auc = roc_auc_score(y_val, y_proba)
            aucs.append(auc)

        mean_auc = np.mean(aucs)

        mlflow.log_metric("roc_auc", mean_auc)

        return mean_auc



def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def main():

    with mlflow.start_run(run_name="Final_Model"):

        X, y = load_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS)

        best_params = study.best_params
        best_params.update({
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "tree_method": "hist"
        })

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_auc", study.best_value)

        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)

        y_proba_test = final_model.predict_proba(X_test)[:, 1]
        best_threshold, best_f1 = find_best_threshold(y_test, y_proba_test)

        mlflow.log_metric("test_f1", best_f1)
        mlflow.log_param("decision_threshold", best_threshold)

  
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(final_model, max_num_features=15)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()

        mlflow.log_artifact("feature_importance.png")


        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            registered_model_name="Fraud_XGBoost_Model"
        )

        print("Training completed and logged to MLflow")



if __name__ == "__main__":
    main()
