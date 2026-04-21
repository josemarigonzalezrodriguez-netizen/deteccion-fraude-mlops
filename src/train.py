import os
import pickle
import mlflow
import mlflow.lightgbm
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Fraude_Bancario")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_experiment(params, run_name, tags=None):
    df = pd.read_csv("data/raw/creditcard.csv")
    X = df.drop(columns="Class")
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        model = LGBMClassifier(**params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "auc_roc":   roc_auc_score(y_test, y_proba),
            "f1_score":  f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
        }

        mlflow.log_metric("auc_roc",   metrics["auc_roc"])
        mlflow.log_metric("f1_score",  metrics["f1_score"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall",    metrics["recall"])

        mlflow.lightgbm.log_model(model, artifact_path="model")

        print(f"\n[{run_name}]")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  F1:        {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        return run_name, metrics


def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 700),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth":        trial.suggest_int("max_depth", 6, 12),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 127),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 20, 100),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 2.0),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    _, metrics = run_experiment(
        params,
        run_name=f"Optuna_trial_{trial.number}",
        tags={"phase": "optuna", "trial": str(trial.number)}
    )
    return metrics["auc_roc"]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    print("\n=== MEJOR RESULTADO OPTUNA ===")
    print(f"AUC:    {study.best_value:.4f}")
    print(f"Params: {study.best_params}")