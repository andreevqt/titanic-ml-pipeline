from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    TRAIN_DATA_PATH, TEST_DATA_PATH,
    REPORTS_DIR, MLFLOW_TRACKING_URI, FLAML_TIME_BUDGET,
)
from src.etl import run_etl
from src.train import run_training
from src.evaluate import run_evaluation
from src.monitoring import run_monitoring


def run_pipeline(
    raw_path: str = RAW_DATA_PATH,
    processed_path: str = PROCESSED_DATA_PATH,
    train_path: str = TRAIN_DATA_PATH,
    test_path: str = TEST_DATA_PATH,
    reports_dir: str = REPORTS_DIR,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    time_budget: int = FLAML_TIME_BUDGET,
) -> None:
    print("ETL:")
    run_etl(raw_path, processed_path, train_path, test_path)

    print("Training:")
    result = run_training(train_path, test_path, tracking_uri, time_budget)
    print(
        f"Best: {result['estimator'].__class__.__name__}, "
        f"ROC-AUC: {result['val_roc_auc']:.4f}, "
        f"Accuracy: {result['val_accuracy']:.4f}, "
        f"F1: {result['val_f1']:.4f}"
    )

    print("Evaluation:")
    run_evaluation(
        result["estimator"],
        result["X_test"],
        result["y_test"],
        result["y_pred"],
        result["y_prob"],
        reports_dir,
    )

    print("Monitoring:")
    drift_path = run_monitoring(train_path, test_path, reports_dir)
    print(f"Drift report: {drift_path}")

    print("Done")


if __name__ == "__main__":
    run_pipeline()
