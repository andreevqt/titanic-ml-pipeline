import pandas as pd
import psutil
import mlflow
import mlflow.sklearn
from flaml import AutoML
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from src.config import (
    FLAML_TIME_BUDGET, RANDOM_SEED,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    TRAIN_DATA_PATH, TEST_DATA_PATH,
)


def run_training(
    train_path: str = TRAIN_DATA_PATH,
    test_path: str = TEST_DATA_PATH,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    time_budget: int = FLAML_TIME_BUDGET,
) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Survived"])
    y_train = train_df["Survived"]
    X_test = test_df.drop(columns=["Survived"])
    y_test = test_df["Survived"]

    with mlflow.start_run() as run:
        automl = AutoML()
        automl.fit(
            X_train, y_train,
            task="classification",
            time_budget=time_budget,
            metric="roc_auc",
            seed=RANDOM_SEED,
            verbose=0,
        )

        y_pred = automl.predict(X_test)
        y_prob = automl.predict_proba(X_test)[:, 1]

        val_roc_auc = roc_auc_score(y_test, y_prob)
        val_accuracy = accuracy_score(y_test, y_pred)
        val_f1 = f1_score(y_test, y_pred)

        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_param("time_budget", time_budget)
        for k, v in automl.best_config.items():
            mlflow.log_param(k, v)

        mlflow.log_metric("val_roc_auc", val_roc_auc)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("cpu_percent", psutil.cpu_percent(interval=1))
        mlflow.log_metric("memory_mb", psutil.virtual_memory().used / 1024 / 1024)

        mlflow.sklearn.log_model(automl.model, "model")
        run_id = run.info.run_id

    return {
        "estimator": automl.model.estimator,
        "run_id": run_id,
        "val_roc_auc": val_roc_auc,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
